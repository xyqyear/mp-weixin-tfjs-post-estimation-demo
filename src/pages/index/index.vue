<template>
  <view class="content">
    <button class="choose" @click="chooseVideo" :disabled="!loaded">
      选择视频
    </button>
    <text v-if="!loaded">正在加载模型</text>
    <canvas type="2d" id="canvas" style="width: 100vw; height: 90vh"></canvas>
  </view>
</template>

<script>
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";

const LINES = [
  // nose - left eye
  [0, 1],
  // nose - right eye
  [0, 2],
  // left eye - left ear
  [1, 3],
  // right eye - right ear
  [2, 4],
  // left shoulder - right shoulder
  [5, 6],
  // left shoulder - left elbow
  [5, 7],
  // left elbow - left wrist
  [7, 9],
  // right shoulder - right elbow
  [6, 8],
  // right elbow - right wrist
  [8, 10],
  // left shoulder - left hip
  [5, 11],
  // right shoulder - right hip
  [6, 12],
  // left hip - right hip
  [11, 12],
  // left hip - left knee
  [11, 13],
  // left knee - left ankle
  [13, 15],
  // right hip - right knee
  [12, 14],
  // right knee - right ankle
  [14, 16],
];

let detector;

export default {
  data() {
    return {
      loaded: false,
    };
  },
  async onLoad() {
    const fs = wx.getFileSystemManager();
    // wrap fs.readFile into promise
    const readFile = (filePath, encoding) => {
      return new Promise((resolve, reject) => {
        fs.readFile({
          filePath,
          encoding,
          success: (res) => {
            resolve(res.data);
          },
          fail: (err) => {
            reject(err);
          },
        });
      });
    };

    const writeFile = (filePath, data, encoding) => {
      return new Promise((resolve, reject) => {
        fs.writeFile({
          filePath,
          data,
          encoding,
          success: (res) => {
            resolve(res);
          },
          fail: (err) => {
            reject(err);
          },
        });
      });
    };

    let modelTopology;
    let weightSpecs;
    let weightData;

    // maybe we can use Promise.all to open and load all the files at once
    // but not sure if wechat is happy about that
    try {
      modelTopology = JSON.parse(
        await readFile(`${wx.env.USER_DATA_PATH}/modelTopology.json`, "utf8")
      );
      weightSpecs = JSON.parse(
        await readFile(`${wx.env.USER_DATA_PATH}/weightSpecs.json`, "utf8")
      );
      weightData = await readFile(
        `${wx.env.USER_DATA_PATH}/weightData.bin`,
        undefined
      );
    } catch (_) {
      const requestFile = (url) => {
        return new Promise((resolve, reject) => {
          uni.request({
            url: url,
            responseType: "arraybuffer",
            success: (res) => {
              resolve(res.data);
            },
            fail: (err) => {
              reject(err);
            },
          });
        });
      };

      const [modelTopology, weightSpecs, weightData] = await Promise.all([
        requestFile(
          "https://mp.muzi.fun/resources/ml-models/movenet-lightning-int8/modelTopology.json"
        ),
        requestFile(
          "https://mp.muzi.fun/resources/ml-models/movenet-lightning-int8/weightSpecs.json"
        ),
        requestFile(
          "https://mp.muzi.fun/resources/ml-models/movenet-lightning-int8/weightData.bin"
        ),
      ]);

      await writeFile(
        `${wx.env.USER_DATA_PATH}/modelTopology.json`,
        JSON.stringify(modelTopology),
        "utf8"
      );

      await writeFile(
        `${wx.env.USER_DATA_PATH}/weightSpecs.json`,
        JSON.stringify(weightSpecs),
        "utf8"
      );

      await writeFile(
        `${wx.env.USER_DATA_PATH}/weightData.bin`,
        weightData,
        undefined
      );
    }

    detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        modelUrl: tf.io.fromMemory({
          modelTopology: modelTopology,
          weightSpecs: weightSpecs,
          weightData: weightData,
        }),
      }
    );
    this.loaded = true;
  },
  methods: {
    shareFile: function (filename) {
      wx.shareFileMessage({
        filePath: `${wx.env.USER_DATA_PATH}/${filename}`,
        success: (res) => {
          console.log(res);
        },
        fail: (err) => {
          console.log(err);
        },
      });
    },
    chooseVideo: async function () {
      const canvas = await new Promise((resolve, reject) => {
        wx.createSelectorQuery()
          .select("#canvas")
          .fields({ node: true, size: true })
          .exec((res) => {
            resolve(res[0].node);
          });
      });
      const canvasCtx = canvas.getContext("2d");

      // tempFilePath, duration
      const videoFileInfo = await wx.chooseMedia({
        count: 1,
        mediaType: ["video"],
      });

      const videoDecoder = wx.createVideoDecoder();
      let ended = false;
      let currentPts;
      videoDecoder.on("ended", () => {
        console.log("ended");
        ended = true;
      });

      videoDecoder.on("start", async () => {
        let startTime;
        let frameData;
        while (true) {
          frameData = videoDecoder.getFrameData();
          if (frameData) {
            startTime = new Date().getTime();
            break;
          }
        }
        const estimateLoop = async (frameData) => {
          const pixels = tf.slice(
            tf.tensor3d(new Uint8Array(frameData.data), [
              frameData.height,
              frameData.width,
              4,
            ]),
            [0, 0, 0],
            [-1, -1, 3]
          );
          const poses = await detector.estimatePoses(
            pixels,
            {},
            frameData.pkPts
          );

          // draw image and pose on canvas
          await new Promise((resolve) => canvas.requestAnimationFrame(resolve));
          canvas.height = frameData.height;
          canvas.width = frameData.width;
          const imageData = canvas.createImageData(
            frameData.data,
            frameData.width,
            frameData.height
          );
          canvasCtx.putImageData(imageData, 0, 0);

          const keypoints = poses[0].keypoints;
          for (let keypoint of keypoints) {
            const x = keypoint.x;
            const y = keypoint.y;
            canvasCtx.beginPath();
            canvasCtx.arc(x, y, 5, 0, 2 * Math.PI);
            canvasCtx.fillStyle = "red";
            canvasCtx.fill();
          }

          // draw the lines between keypoints using LINES variable
          for (let line of LINES) {
            const keypoint1 = keypoints[line[0]];
            const keypoint2 = keypoints[line[1]];
            const x1 = keypoint1.x;
            const y1 = keypoint1.y;
            const x2 = keypoint2.x;
            const y2 = keypoint2.y;
            canvasCtx.beginPath();
            canvasCtx.moveTo(x1, y1);
            canvasCtx.lineTo(x2, y2);
            canvasCtx.strokeStyle = "cyan";
            canvasCtx.stroke();
          }

          if (!ended) {
            frameData = videoDecoder.getFrameData();
            if (frameData) {
              currentPts = frameData.pkPts;
              estimateLoop(frameData);
            } else {
              console.log(
                "scale:",
                (new Date().getTime() - startTime) / currentPts
              );
            }
          }
        };

        estimateLoop(frameData);
      });

      videoDecoder.start({
        source: videoFileInfo.tempFiles[0].tempFilePath,
        abortAudio: true,
      });
    },
  },
};
</script>

<style>
</style>
