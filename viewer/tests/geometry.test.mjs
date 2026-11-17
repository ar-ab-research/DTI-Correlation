import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { check } from "../scripts/check-public.mjs";
import {
  VIEWS, fitRect, fromByte, screenToVoxel, sliceCoord, sliceProfile, voxelToMm, voxelToScreen,
  windowRange,
} from "../src/geometry.mjs";

const dims = [10, 20, 30];

test("each view looks along one axis and draws the other two", () => {
  for (const view of Object.values(VIEWS)) {
    assert.deepEqual([view.axis, view.h, view.v].sort(), [0, 1, 2]);
  }
});

test("the slice is fitted with its own shape and centred", () => {
  assert.deepEqual(fitRect(200, 100, 10, 10), { x: 50, y: 0, w: 100, h: 100 });
  assert.deepEqual(fitRect(100, 300, 10, 20), { x: 0, y: 50, w: 100, h: 200 });
  assert.deepEqual(fitRect(0, 100, 10, 10), { x: 0, y: 0, w: 0, h: 0 });
});

test("axial and coronal put the left of the subject on the right of the screen", () => {
  // x grows towards the right of the subject, so the left edge of the screen is the highest x
  assert.equal(screenToVoxel("axial", 0.01, 0.5, [0, 0, 5], dims)[0], 9);
  assert.equal(screenToVoxel("axial", 0.99, 0.5, [0, 0, 5], dims)[0], 0);
  assert.equal(screenToVoxel("coronal", 0.01, 0.5, [0, 3, 0], dims)[0], 9);
  assert.equal(VIEWS.axial.labels.left, "R");
});

test("up on screen is forward on axial and up on the other two", () => {
  assert.equal(screenToVoxel("axial", 0.5, 0.01, [0, 0, 5], dims)[1], 19);
  assert.equal(screenToVoxel("coronal", 0.5, 0.01, [0, 3, 0], dims)[2], 29);
  assert.equal(screenToVoxel("sagittal", 0.99, 0.99, [4, 0, 0], dims)[1], 19);
  assert.equal(screenToVoxel("sagittal", 0.99, 0.99, [4, 0, 0], dims)[2], 0);
});

test("a click keeps the position along the viewing axis and stays inside the volume", () => {
  assert.equal(screenToVoxel("axial", 0.5, 0.5, [1, 2, 17], dims)[2], 17);
  assert.deepEqual(screenToVoxel("axial", 1, 0, [0, 0, 0], dims), [0, 19, 0]);
  assert.deepEqual(screenToVoxel("axial", 0, 1, [0, 0, 0], dims), [9, 0, 0]);
});

test("screen to voxel and back lands in the same voxel, in every view", () => {
  for (const name of Object.keys(VIEWS)) {
    for (const voxel of [[0, 0, 0], [9, 19, 29], [3, 11, 7]]) {
      const { fx, fy } = voxelToScreen(name, voxel, dims);
      assert.deepEqual(screenToVoxel(name, fx, fy, voxel, dims), voxel, name);
    }
  }
});

test("slice coordinate is the centre of the slice", () => {
  assert.equal(sliceCoord(0, 10), 0.05);
  assert.equal(sliceCoord(9, 10), 0.95);
});

test("the tract profile counts one tract's voxels per slice along each axis", () => {
  const small = [2, 3, 4];
  const codes = new Uint8Array(24);
  const set = (x, y, z, code) => { codes[x + 2 * (y + 3 * z)] = code; };
  set(0, 0, 0, 1); set(1, 0, 0, 1); set(1, 2, 3, 1); set(0, 1, 0, 2);
  assert.deepEqual([...sliceProfile(codes, small, 2, 1)], [2, 0, 0, 1]);
  assert.deepEqual([...sliceProfile(codes, small, 1, 1)], [2, 0, 1]);
  assert.deepEqual([...sliceProfile(codes, small, 0, 1)], [1, 2]);
  // the other tract is counted on its own
  assert.deepEqual([...sliceProfile(codes, small, 2, 2)], [1, 0, 0, 0]);
});

test("brightness and contrast", () => {
  assert.deepEqual(windowRange(0.5, 1), { low: 0, span: 1 });
  assert.deepEqual(windowRange(0.5, 0.5), { low: 0.25, span: 0.5 });
  assert.ok(windowRange(0.5, 0).span > 0);
});

test("position in mm is measured from the centre of the volume", () => {
  assert.deepEqual(voxelToMm([5, 10, 15], dims, 1), [0.5, 0.5, 0.5]);
  assert.deepEqual(voxelToMm([0, 0, 0], dims, 2), [-9, -19, -29]);
});

test("a stored byte turns back into the value it stands for", () => {
  assert.equal(fromByte(255, 1), 1);
  assert.equal(fromByte(0, 3), 0);
  assert.equal(fromByte(85, 3), 1);
});

test("a scan that is not on the public list blocks the deploy", () => {
  const dir = mkdtempSync(join(tmpdir(), "viewer-"));
  const data = join(dir, "data");
  mkdirSync(join(data, "patient01"), { recursive: true });
  mkdirSync(join(data, "sub-secret"), { recursive: true });
  const allow = join(dir, "public-subjects.json");
  writeFileSync(allow, JSON.stringify({ subjects: { patient01: "public" } }));
  // the menu lists only the public one, the stray folder must still be caught
  writeFileSync(join(data, "index.json"), JSON.stringify({ subjects: ["patient01"] }));
  assert.deepEqual(check(data, allow).blocked, ["sub-secret"]);
  writeFileSync(allow, JSON.stringify({ subjects: { patient01: "public", "sub-secret": "x" } }));
  assert.deepEqual(check(data, allow).blocked, []);
  assert.deepEqual(check(join(dir, "missing"), allow).present, []);
});
