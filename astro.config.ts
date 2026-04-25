import { defineConfig, envField } from "astro/config";
import tailwindcss from "@tailwindcss/vite";
import sitemap from "@astrojs/sitemap";
import remarkToc from "remark-toc";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkCollapse from "remark-collapse";
import mdx from "@astrojs/mdx";
import expressiveCode from "astro-expressive-code";
import { pluginShiki, pluginFrames } from "astro-expressive-code";
import rehypeExpressiveCode from "rehype-expressive-code";
import { SITE } from "./src/config";

// https://astro.build/config
export default defineConfig({
  site: SITE.website,
  integrations: [
    expressiveCode({
      // Explicitly list plugins to exclude pluginTextMarkers (which causes
      // spurious line highlights on comment separators like # -----)
      plugins: [pluginShiki(), pluginFrames()],
      themes: ["night-owl", "min-light"],
      defaultProps: {
        wrap: false,
      },
    }),
    sitemap({
      filter: page => SITE.showArchives || !page.endsWith("/archives"),
    }),
    mdx({
      remarkPlugins: [
        remarkMath,
        remarkToc,
        [remarkCollapse, { test: "Table of contents" }],
      ],
      rehypePlugins: [
        rehypeKatex,
        [
          rehypeExpressiveCode,
          {
            plugins: [pluginShiki(), pluginFrames()],
            themes: ["night-owl", "min-light"],
          },
        ],
      ],
    }),
  ],
  markdown: {
    remarkPlugins: [
      remarkMath,
      remarkToc,
      [remarkCollapse, { test: "Table of contents" }],
    ],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      themes: { light: "min-light", dark: "night-owl" },
      defaultColor: false,
      wrap: false,
    },
  },
  vite: {
    plugins: [tailwindcss()],
    optimizeDeps: {
      exclude: ["@resvg/resvg-js"],
    },
  },
  image: {
    responsiveStyles: true,
    layout: "constrained",
  },
  env: {
    schema: {
      PUBLIC_GOOGLE_SITE_VERIFICATION: envField.string({
        access: "public",
        context: "client",
        optional: true,
      }),
    },
  },
});
