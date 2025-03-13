---
---

# Model Card


Here is a detailed description of how CogKit supports models.

<p>All training requirements <b style="color: red;">MUST</b> be strictly followed as specified in the table below, including <b style="color: red;">Resolution</b>, <b style="color: red;">Number of Frames</b>, <b style="color: red;">Prompt Token Limit</b>, and <b style="color: red;">Video Length</b> requirements.</p>

## CogVideo

<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">Model Name</th>
    <th style="text-align: center;">CogVideoX1.5-5B (Latest)</th>
    <th style="text-align: center;">CogVideoX1.5-5B-I2V (Latest)</th>
    <th style="text-align: center;">CogVideoX-2B</th>
    <th style="text-align: center;">CogVideoX-5B</th>
    <th style="text-align: center;">CogVideoX-5B-I2V</th>
  </tr>
  <tr>
    <td style="text-align: center;">Release Date</td>
    <th style="text-align: center;">November 8, 2024</th>
    <th style="text-align: center;">November 8, 2024</th>
    <th style="text-align: center;">August 6, 2024</th>
    <th style="text-align: center;">August 27, 2024</th>
    <th style="text-align: center;">September 19, 2024</th>
  </tr>
  <tr>
    <td style="text-align: center;">Resolution</td>
    <td colspan="1" style="text-align: center;">1360 * 768</td>
    <td colspan="1" style="text-align: center;"> Min(W, H) = 768 <br> 768 ≤ Max(W, H) ≤ 1360 <br> Max(W, H) % 16 = 0 </td>
    <td colspan="3" style="text-align: center;">720 * 480</td>
  </tr>
  <tr>
    <td style="text-align: center;">Number of Frames</td>
    <td colspan="2" style="text-align: center;">Should be <b>16N + 1</b> where N <= 10 (default 81)</td>
    <td colspan="3" style="text-align: center;">Should be <b>8N + 1</b> where N <= 6 (default 49)</td>
  </tr>
  <tr>
    <td style="text-align: center;">Prompt Language</td>
    <td colspan="5" style="text-align: center;">English</td>
  </tr>
  <tr>
    <td style="text-align: center;">Prompt Token Limit</td>
    <td colspan="2" style="text-align: center;">224 Tokens (T5)</td>
    <td colspan="3" style="text-align: center;">226 Tokens (T5)</td>
  </tr>
  <tr>
    <td style="text-align: center;">Video Length</td>
    <td colspan="2" style="text-align: center;">5 seconds or 10 seconds</td>
    <td colspan="3" style="text-align: center;">6 seconds</td>
  </tr>
  <tr>
    <td style="text-align: center;">Frame Rate</td>
    <td colspan="2" style="text-align: center;">16 frames / second </td>
    <td colspan="3" style="text-align: center;">8 frames / second </td>
  </tr>
  <tr>
    <td style="text-align: center;">Download Link (Diffusers)</td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX1.5-5B">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX1.5-5B">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX1.5-5B-I2V">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX1.5-5B-I2V">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-2b">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-2b">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-2b">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-5b">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-5b">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-5b">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-5b-I2V">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-5b-I2V">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-5b-I2V">🟣 WiseModel</a></td>
  </tr>
</table>


## CogView


<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">Model Name</th>
    <th style="text-align: center;">CogView4-6B (Latest)</th>
  </tr>
  <tr>
    <td style="text-align: center;">Release Date</td>
    <th style="text-align: center;">March 4, 2025</th>
  </tr>
  <tr>
    <td style="text-align: center;">Resolution</td>
    <td style="text-align: center;">512 ≤ (W, H) ≤ 2048 <br> H * W ≤ 2^{21} <br> Max(W, H) % 32 = 0 </td>
  </tr>
  <tr>
    <td style="text-align: center;">Prompt Language</td>
    <td style="text-align: center;">English，简体中文</td>
  </tr>
  <tr>
    <td style="text-align: center;">Prompt Token Limit</td>
    <td style="text-align: center;">1024 Tokens (GLM-4-9B)</td>
  </tr>
  <tr>
    <td style="text-align: center;">Download Link (Diffusers)</td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogView4-6B">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogView4-6B">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogView4-6B">🟣 WiseModel</a></td>
</table>

