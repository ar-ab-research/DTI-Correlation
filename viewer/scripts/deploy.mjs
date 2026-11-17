// publishes dist/ to the gh-pages branch, with no history, as the only thing on it.
//
// the gh-pages tool starts the branch from a checkout of main and only clears files whose
// names do not start with a dot. main's .gitignore would then stay behind on gh-pages and
// keep data/, the scans the site shows, out of every deploy. so main's two dotfiles are
// taken out before anything is added.
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import ghpages from "gh-pages";

const dist = join(dirname(fileURLToPath(import.meta.url)), "..", "dist");

ghpages.publish(dist, {
  history: false,
  nojekyll: true,
  dotfiles: true,
  message: "rebuild the viewer site",
  beforeAdd: (git) => git.rm([".gitignore", ".gitattributes"]),
}, (error) => {
  if (error) {
    console.error(`not deployed: ${error.message}`);
    process.exit(1);
  }
  console.log("published");
});
