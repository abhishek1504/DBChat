import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The proxy forwards /api/* to FastAPI on :8000 during development,
// so the browser sees a single origin and CORS never gets in the way.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
