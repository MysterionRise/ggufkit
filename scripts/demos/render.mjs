// Render the demos recorded by scripts/make_demos.py into videos, GIFs and posters.
//
//   node scripts/demos/render.mjs DEMOS_JSON [NAME ...] [--formats=vertical,feed,readme] [--keep-frames]
//
// Formats, from scenes.html (vertical and readme unless --formats says otherwise):
//   vertical  1080x1920, 30 fps -> build/demos/NAME.mp4 and NAME-cover.png
//   feed      1080x1350, 30 fps -> build/demos/NAME-feed.mp4 and NAME-feed-cover.png (4:5)
//   readme     800x1000, 15 fps -> assets/demos/NAME.gif and assets/demos/NAME.png
// Every MP4 also gets an .srt with its on-screen text as captions (NAME.srt, NAME-feed.srt).
// Needs Playwright (with a Chromium it can launch) and ffmpeg on PATH.

import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

// Through require, so a global Playwright install is found via NODE_PATH.
const { chromium } = createRequire(import.meta.url)("playwright");

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..", "..");
const SCENES = pathToFileURL(join(HERE, "scenes.html")).href;
// layout: the scenes.html layout to use; out: what to encode the frames into.
const FORMATS = {
  vertical: { width: 1080, height: 1920, scale: 1, fps: 30, layout: "vertical", out: "mp4", suffix: "" },
  feed: { width: 1080, height: 1350, scale: 1, fps: 30, layout: "readme", out: "mp4", suffix: "-feed" },
  readme: { width: 1080, height: 1350, scale: 800 / 1080, fps: 15, layout: "readme", out: "gif", suffix: "" },
};
const DEFAULT_FORMATS = ["vertical", "readme"];

const args = process.argv.slice(2);
const flags = new Set(args.filter((a) => a.startsWith("--") && !a.includes("=")));
const formatArg = args.find((a) => a.startsWith("--formats="));
const positional = args.filter((a) => !a.startsWith("--"));
const [dataPath, ...only] = positional;
if (!dataPath) {
  console.error("usage: node render.mjs DEMOS_JSON [NAME ...] [--formats=vertical,feed,readme]");
  process.exit(2);
}
const data = JSON.parse(readFileSync(dataPath, "utf8"));
const names = only.length ? only : Object.keys(data).filter((k) => k !== "tools");
const formats = formatArg ? formatArg.split("=")[1].split(",") : DEFAULT_FORMATS;
const unknown = formats.filter((f) => !(f in FORMATS));
if (unknown.length) {
  console.error(`unknown format(s): ${unknown.join(", ")}`);
  process.exit(2);
}

function ffmpeg(...argv) {
  const r = spawnSync("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", ...argv], { stdio: "inherit" });
  if (r.status !== 0) throw new Error(`ffmpeg failed: ${argv.join(" ")}`);
}

const mb = (path) => (statSync(path).size / 1e6).toFixed(2);

function srtTime(seconds) {
  const ms = Math.round(seconds * 1000);
  const pad = (n, w = 2) => String(n).padStart(w, "0");
  return `${pad(Math.floor(ms / 3600000))}:${pad(Math.floor(ms / 60000) % 60)}:${pad(Math.floor(ms / 1000) % 60)},${pad(ms % 1000, 3)}`;
}

function srt(cues) {
  return cues.map((c, i) => `${i + 1}\n${srtTime(c.start)} --> ${srtTime(c.end)}\n${c.text}\n`).join("\n");
}

const browser = await chromium.launch();
try {
  for (const format of formats) {
    const f = FORMATS[format];
    const context = await browser.newContext({
      viewport: { width: f.width, height: f.height },
      deviceScaleFactor: f.scale,
    });
    for (const name of names) {
      const page = await context.newPage();
      await page.goto(SCENES);
      const meta = await page.evaluate(([n, d, l]) => window.setup(n, d, l), [name, data[name], f.layout]);
      await page.evaluate(() => window.ready());

      const frames = join(ROOT, "build", "demos", "frames", `${name}-${format}`);
      rmSync(frames, { recursive: true, force: true });
      mkdirSync(frames, { recursive: true });
      const count = Math.round(meta.duration * f.fps);
      for (let i = 0; i < count; i++) {
        await page.evaluate((t) => window.renderAt(t), i / f.fps);
        await page.screenshot({ path: join(frames, `${String(i).padStart(5, "0")}.png`) });
      }
      await page.evaluate((t) => window.renderAt(t), meta.poster);

      if (f.out === "mp4") {
        const out = join(ROOT, "build", "demos");
        await page.screenshot({ path: join(out, `${name}${f.suffix}-cover.png`) });
        const captions = join(out, `${name}${f.suffix}.srt`);
        writeFileSync(captions, srt(await page.evaluate(() => window.captions())));
        const mp4 = join(out, `${name}${f.suffix}.mp4`);
        ffmpeg(
          "-framerate", String(f.fps), "-i", join(frames, "%05d.png"),
          "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
          "-map", "0:v", "-map", "1:a", "-shortest",
          "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
          "-profile:v", "high", "-r", String(f.fps),
          "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", mp4,
        );
        console.log(`${name}: ${mp4} (${mb(mp4)} MB, ${count} frames)`);
      } else {
        const out = join(ROOT, "assets", "demos");
        mkdirSync(out, { recursive: true });
        await page.screenshot({ path: join(out, `${name}.png`) });
        const gif = join(out, `${name}.gif`);
        ffmpeg(
          "-framerate", String(f.fps), "-i", join(frames, "%05d.png"),
          "-vf", "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
          "-loop", "0", gif,
        );
        console.log(`${name}: ${gif} (${mb(gif)} MB, ${count} frames)`);
      }
      if (!flags.has("--keep-frames")) rmSync(frames, { recursive: true, force: true });
      await page.close();
    }
    await context.close();
  }
} finally {
  await browser.close();
}
