import type { NextConfig } from "next";

/**
 * UCRA project site — static export config.
 *
 * - `output: "export"` makes `npm run build` emit a pure-static `out/` folder
 *   you can host anywhere (GitHub Pages, Netlify, any static server).
 * - If you deploy to a GitHub Pages PROJECT site
 *   (https://<user>.github.io/<repo>/), uncomment `basePath` and set it to
 *   your repository name, e.g. basePath: "/UCRA".
 * - If you deploy to Vercel or a user site (https://<user>.github.io/),
 *   leave basePath commented out.
 */
const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  // basePath: "/UCRA",
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
};

export default nextConfig;
