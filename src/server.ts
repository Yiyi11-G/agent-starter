import { createWorkersAI } from "workers-ai-provider";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  streamText,
  tool
} from "ai";
import { z } from "zod";

export class ChatAgent extends AIChatAgent<Env> {
  maxPersistedMessages = 100;
  chatRecovery = true;
  waitForMcpConnections = true;

  onStart() {
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: {
              "content-type": "text/html"
            },
            status: 200
          });
        }

        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          {
            headers: {
              "content-type": "text/plain"
            },
            status: 400
          }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(
    _onFinish: unknown,
    options?: OnChatMessageOptions
  ) {
    const mcpTools = this.mcp.getAITools();

    const workersai = createWorkersAI({
      binding: this.env.AI
    });

    const result = streamText({
      model: workersai("@cf/zai-org/glm-4.7-flash"),

      system: `You are a helpful AI assistant.

You can:
- Have natural conversations with the user.
- Understand questions in multiple languages.
- Understand images provided by the user.
- Check weather when the user asks about weather.
- Get the user's timezone when the user asks about their timezone or local time.
- Perform mathematical calculations.
- Schedule tasks.
- Use MCP tools when appropriate.

IMPORTANT RULES:

1. For normal greetings such as "hi", "hello", "你好", "hey", etc.,
   respond naturally and DO NOT call any tools.

2. Only call a tool when the user's request actually requires it.

3. Do not call getUserTimezone just because the user says hello.

4. Do not repeatedly call the same tool.

5. After receiving a tool result, answer the user naturally using that result.

6. Never expose internal tool-call JSON or implementation details.

7. If a tool fails, do not repeatedly retry it.

8. Keep normal conversational responses concise and natural.

${getSchedulePrompt({
  date: new Date()
})}

If the user asks to schedule a task, use the scheduleTask tool.`,

      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages",
        reasoning: "before-last-message"
      }),

      tools: {
        ...mcpTools,

        getWeather: tool({
          description:
            "Get the current weather for a city. Only call this tool when the user explicitly asks about weather.",

          inputSchema: z.object({
            city: z.string().describe("City name")
          }),

          execute: async ({ city }) => {
            const conditions = [
              "sunny",
              "cloudy",
              "rainy",
              "snowy"
            ];

            const temperature =
              Math.floor(Math.random() * 30) + 5;

            return {
              city,
              temperature,
              condition:
                conditions[
                  Math.floor(Math.random() * conditions.length)
                ],
              unit: "celsius"
            };
          }
        }),

        getUserTimezone: tool({
          description:
            "Get the user's local timezone and local time from their browser. Only call this when the user asks about their timezone or local time.",

          inputSchema: z.object({})
        }),

        calculate: tool({
          description:
            "Perform a mathematical calculation with two numbers.",

          inputSchema: z.object({
            a: z.number(),
            b: z.number(),
            operator: z.enum([
              "+",
              "-",
              "*",
              "/",
              "%"
            ])
          }),

          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,

          execute: async ({ a, b, operator }) => {
            if (operator === "/" && b === 0) {
              return {
                error: "Division by zero"
              };
            }

            const ops: Record<
              string,
              (x: number, y: number) => number
            > = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y
            };

            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b)
            };
          }
        }),

        scheduleTask: tool({
          description:
            "Schedule a task to be executed at a later time.",

          inputSchema: scheduleSchema,

          execute: async ({ when, description }) => {
            if (when.type === "no-schedule") {
              return "Not a valid schedule input.";
            }

            const input =
              when.type === "scheduled"
                ? when.date
                : when.type === "delayed"
                  ? when.delayInSeconds
                  : when.type === "cron"
                    ? when.cron
                    : null;

            if (!input) {
              return "Invalid schedule type.";
            }

            try {
              this.schedule(
                input,
                "executeTask",
                description,
                {
                  idempotent: true
                }
              );

              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${
                error instanceof Error
                  ? error.message
                  : String(error)
              }`;
            }
          }
        }),

        getScheduledTasks: tool({
          description: "List all scheduled tasks.",

          inputSchema: z.object({}),

          execute: async () => {
            const tasks = this.getSchedules();

            return tasks.length > 0
              ? tasks
              : "No scheduled tasks found.";
          }
        }),

        cancelScheduledTask: tool({
          description:
            "Cancel a scheduled task by its ID.",

          inputSchema: z.object({
            taskId: z.string()
          }),

          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);

              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${
                error instanceof Error
                  ? error.message
                  : String(error)
              }`;
            }
          }
        })
      },

      stopWhen: stepCountIs(8),

      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(
    description: string,
    _task: Schedule<string>
  ) {
    console.log(
      `Executing scheduled task: ${description}`
    );

    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", {
        status: 404
      })
    );
  }
} satisfies ExportedHandler<Env>;
