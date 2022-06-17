<script>
import * as fetchWechat from "fetch-wechat";
import * as tf from "@tensorflow/tfjs-core";
import * as webgl from "@tensorflow/tfjs-backend-webgl";

const plugin = requirePlugin("tfjsPlugin");

export default {
  onLaunch: function () {
    console.log("App Launch");
    plugin.configPlugin({
      // polyfill fetch function
      fetchFunc: fetchWechat.fetchFunc(),
      // inject tfjs runtime
      tf,
      // inject webgl backend
      webgl,
      // provide webgl canvas
      canvas: wx.createOffscreenCanvas(),
    });
  },
  onShow: function () {
    console.log("App Show");
  },
  onHide: function () {
    console.log("App Hide");
  },
};
</script>

<style>
/*每个页面公共css */
</style>
