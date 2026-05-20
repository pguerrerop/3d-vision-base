import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

function envPort(value: string | undefined, fallback: number): number {
  const port = Number(value);
  return Number.isInteger(port) && port > 0 && port <= 65535 ? port : fallback;
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const frontendPort = envPort(env.VITE_PORT ?? env.PORT, 5173);
  const apiPort = envPort(env.VITE_API_PORT ?? env.API_PORT, 8000);
  const apiHost = env.VITE_API_HOST ?? env.API_HOST ?? "localhost";
  const apiOrigin = env.VITE_API_ORIGIN ?? `http://${apiHost}:${apiPort}`;

  return {
    plugins: [react()],
    server: {
      port: frontendPort,
      proxy: {
        "/api": apiOrigin
      }
    }
  };
});
