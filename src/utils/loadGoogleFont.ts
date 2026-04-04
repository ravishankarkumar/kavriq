import { readFileSync } from "fs";
import { resolve } from "path";

async function loadGoogleFont(
  font: string,
  text: string,
  weight: number
): Promise<ArrayBuffer> {
  const API = `https://fonts.googleapis.com/css2?family=${font}:wght@${weight}&text=${encodeURIComponent(text)}`;

  const css = await (
    await fetch(API, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; de-at) AppleWebKit/533.21.1 (KHTML, like Gecko) Version/5.0.5 Safari/533.21.1",
      },
    })
  ).text();

  const resource = css.match(
    /src: url$(.+?)$ format$'(opentype|truetype)'$/
  );

  if (!resource) throw new Error("Failed to download dynamic font");

  const res = await fetch(resource[1]);

  if (!res.ok) {
    throw new Error("Failed to download dynamic font. Status: " + res.status);
  }

  return res.arrayBuffer();
}

function loadLocalFallbackFont(weight: 400 | 700): ArrayBuffer {
  const filename =
    weight === 700 ? "KaTeX_Main-Bold.ttf" : "KaTeX_Main-Regular.ttf";
  const fontPath = resolve(`node_modules/katex/dist/fonts/${filename}`);
  const buffer = readFileSync(fontPath);
  return buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength
  ) as ArrayBuffer;
}

async function loadGoogleFonts(
  text: string
): Promise<
  Array<{ name: string; data: ArrayBuffer; weight: number; style: string }>
> {
  const fontsConfig = [
    {
      name: "IBM Plex Mono",
      font: "IBM+Plex+Mono",
      weight: 400,
      style: "normal",
    },
    {
      name: "IBM Plex Mono",
      font: "IBM+Plex+Mono",
      weight: 700,
      style: "bold",
    },
  ];

  const fonts = await Promise.all(
    fontsConfig.map(async ({ name, font, weight, style }) => {
      try {
        const data = await loadGoogleFont(font, text, weight);
        return { name, data, weight, style };
      } catch {
        return null;
      }
    })
  );

  const loaded = fonts.filter(Boolean) as Array<{
    name: string;
    data: ArrayBuffer;
    weight: number;
    style: string;
  }>;

  if (loaded.length === 0) {
    // Fall back to local KaTeX fonts (TTF) so satori always has at least one font
    return [
      { name: "Fallback", data: loadLocalFallbackFont(400), weight: 400, style: "normal" },
      { name: "Fallback", data: loadLocalFallbackFont(700), weight: 700, style: "bold" },
    ];
  }

  return loaded;
}

export default loadGoogleFonts;
