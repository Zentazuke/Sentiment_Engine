import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const target = process.env.SENTIMENT_API_TARGET || "http://127.0.0.1:8787";

export default defineConfig({
  plugins: [react()],
  // Build straight into the engine repo's committed dashboard_dist/, which
  // serve_dashboard.py serves and `git pull` ships to the server (no scp).
  build: {
    outDir: "../dashboard_dist",
    emptyOutDir: true
  },
  server: {
    proxy: {
      "/api": {
        target,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, "")
      }
    }
  }
});
