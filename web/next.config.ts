import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow cross-origin requests to the version API from the MindVault CLI
  async headers() {
    return [
      {
        source: "/api/:path*",
        headers: [
          { key: "Access-Control-Allow-Origin", value: "*" },
          { key: "Access-Control-Allow-Methods", value: "GET" },
        ],
      },
    ];
  },
};

export default nextConfig;
