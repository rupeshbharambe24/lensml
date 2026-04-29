import type { NextConfig } from "next";

const FASTAPI_URL = process.env.LENSML_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // Proxy /api/* and /healthz to the lensml FastAPI server during dev.
  // Keeps SSE same-origin (no CORS preflight) and avoids hard-coding the
  // backend URL in browser code.
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${FASTAPI_URL}/api/:path*` },
      { source: "/healthz", destination: `${FASTAPI_URL}/healthz` },
    ];
  },
};

export default nextConfig;
