import base64
import io
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import aiofiles
import httpx
import magic
from httpx import Timeout
from PIL import Image

from nekro_agent.core import logger
from nekro_agent.core.config import ModelConfigGroup
from nekro_agent.services.agent.creator import ContentSegment, OpenAIChatMessage
from nekro_agent.tools.path_convertor import convert_to_host_path


async def download_image(url: str) -> Image.Image:
    """从 URL 下载图片"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))


def decode_base64_image(base64_string: str) -> Image.Image:
    """解码 Base64 图片"""
    if base64_string.startswith("data:image"):
        base64_string = base64_string.split(",")[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


def split_sprite_sheet(image: Image.Image, rows: int = 4, cols: int = 4) -> List[Image.Image]:
    """将精灵图切割成单独的帧"""
    width, height = image.size
    frame_width = width // cols
    frame_height = height // rows

    frames = []
    for r in range(rows):
        for c in range(cols):
            left = c * frame_width
            upper = r * frame_height
            right = left + frame_width
            lower = upper + frame_height
            frame = image.crop((left, upper, right, lower))
            frames.append(frame)
    return frames


def create_gif_from_frames(
    frames: List[Image.Image],
    output_path: str,
    duration: int = 100,
    transparency_color: Optional[Tuple[int, int, int]] = None,
    tolerance: int = 10,
) -> None:
    """将帧序列保存为 GIF

    Args:
        frames: 图片帧列表
        output_path: 输出路径
        duration: 每帧持续时间 (ms)
        transparency_color: 要作为透明处理的颜色 (R, G, B)
        tolerance: 颜色匹配容差
    """
    if not frames:
        return

    processed_frames = []
    for frame in frames:
        # 确保是 RGBA 模式以便处理透明度
        img = frame.convert("RGBA")

        if transparency_color:
            width, height = img.size
            new_data: List[Tuple[int, int, int, int]] = []
            tr, tg, tb = transparency_color

            # 逐像素处理
            for y in range(height):
                for x in range(width):
                    pixel = img.getpixel((x, y))
                    if isinstance(pixel, tuple) and len(pixel) == 4:
                        r, g, b, a = int(pixel[0]), int(pixel[1]), int(pixel[2]), int(pixel[3])
                        # 简单的容差判断
                        if abs(r - tr) <= tolerance and abs(g - tg) <= tolerance and abs(b - tb) <= tolerance:
                            new_data.append((255, 255, 255, 0))  # 完全透明
                        else:
                            new_data.append((r, g, b, a))

            img.putdata(new_data)

        processed_frames.append(img)

    # 保存 GIF
    processed_frames[0].save(
        output_path,
        save_all=True,
        append_images=processed_frames[1:],
        optimize=False,
        duration=duration,
        loop=0,
        disposal=2,  # 2 表示恢复背景色，有助于处理透明背景动画
    )


def get_average_color(image: Image.Image) -> Tuple[int, int, int]:
    """获取图片的平均颜色"""
    # 缩放到 1x1 获取平均色
    img = image.copy()
    img = img.resize((1, 1), Image.Resampling.LANCZOS)
    color = img.getpixel((0, 0))
    if isinstance(color, int):  # 灰度图
        return (color, color, color)
    if isinstance(color, tuple) and len(color) >= 3:
        return (int(color[0]), int(color[1]), int(color[2]))
    # 兜底
    return (0, 0, 0)


async def prepare_reference_image(
    refer_image: str,
    chat_key: str,
    container_key: str,
) -> str:
    """准备参考图片数据

    Args:
        refer_image: 参考图片路径
        chat_key: 聊天键
        container_key: 容器键

    Returns:
        str: base64 编码的图片数据（格式: data:image/xxx;base64,xxx）
    """
    if not refer_image:
        return ""

    async with aiofiles.open(
        convert_to_host_path(Path(refer_image), chat_key=chat_key, container_key=container_key),
        mode="rb",
    ) as f:
        image_data = await f.read()
        mime_type = magic.from_buffer(image_data, mime=True)
        image_data_b64 = base64.b64encode(image_data).decode("utf-8")

    return f"data:{mime_type};base64,{image_data_b64}"


async def generate_image_via_chat(
    model_group: ModelConfigGroup,
    prompt: str,
    timeout: float = 300.0,
    system_prompt: Optional[str] = None,
    use_system_role: bool = False,
    reference_images: Optional[List[Tuple[str, str]]] = None,
    stream_mode: bool = True,
) -> str:
    """通过聊天接口生成图片的通用方法

    Args:
        model_group: 模型组配置对象，需包含 CHAT_MODEL, API_KEY, BASE_URL
        prompt: 图像生成提示词
        timeout: 请求超时时间（秒），默认 300 秒
        system_prompt: 系统提示词，如果不提供则使用默认值
        use_system_role: 是否使用系统角色（某些模型不支持）
        reference_images: 参考图片列表，格式为 [(base64_data, description), ...]
        stream_mode: 是否使用流式模式，默认 True（推荐，避免长时间超时）

    Returns:
        str: 图片 URL 或 base64 数据（格式: data:image/png;base64,xxx）

    Raises:
        ValueError: 当模型未返回内容或无法提取图片时
        httpx.HTTPError: 当 HTTP 请求失败时
    """
    if system_prompt is None:
        system_prompt = "You are a professional painter. Use your high-quality drawing skills to draw a picture based on the user's description. Just provide the image and do not ask for more information."

    # 构建消息
    msg = OpenAIChatMessage.create_empty("user")

    # 添加参考图片
    if reference_images:
        for img_data, img_desc in reference_images:
            if img_data:
                msg = msg.add(ContentSegment.image_content(img_data))
                if img_desc:
                    msg = msg.add(ContentSegment.text_content(f"{img_desc}\n"))

    # 添加主提示词
    if not use_system_role and system_prompt:
        # 如果不使用系统角色，将系统提示词合并到用户消息中
        full_prompt = f"{system_prompt}\n\n{prompt}"
        msg = msg.add(ContentSegment.text_content(full_prompt))
    else:
        msg = msg.add(ContentSegment.text_content(prompt))

    # 构建消息列表
    messages = []
    if use_system_role and system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(msg.to_dict())

    # 发送请求
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model_group.API_KEY}",
    }

    json_data = {
        "model": model_group.CHAT_MODEL,
        "messages": messages,
        "stream": stream_mode,
    }

    collected_content = ""
    collected_image_data: Optional[str] = None

    async with httpx.AsyncClient(timeout=Timeout(read=timeout, write=timeout, connect=10, pool=10)) as client:
        if stream_mode:
            # 流式请求
            async with client.stream(
                "POST",
                f"{model_group.BASE_URL}/chat/completions",
                headers=headers,
                json=json_data,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue

                    # 处理 SSE 格式
                    if line.startswith("data: "):
                        data_str = line[6:]  # 移除 "data: " 前缀
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            choices = chunk_data.get("choices", [])
                            if not choices:
                                continue

                            delta = choices[0].get("delta", {})

                            # 优先检查 image 字段
                            image_data = delta.get("image")
                            if image_data and isinstance(image_data, list) and image_data:
                                # 取第一张图片的 base64 数据
                                collected_image_data = image_data[0].get("data")
                                if isinstance(collected_image_data, str):
                                    logger.debug(f"找到 image 字段，数据长度: {len(collected_image_data)}")

                            # 收集 content 内容作为备选
                            content_data = delta.get("content")
                            if content_data:
                                collected_content += content_data

                        except json.JSONDecodeError as e:
                            logger.debug(f"解析 JSON 失败: {e}, 数据: {data_str}")
                            continue
        else:
            # 非流式请求
            response = await client.post(
                f"{model_group.BASE_URL}/chat/completions",
                headers=headers,
                json=json_data,
            )
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            if not choices:
                raise ValueError("模型未返回任何内容")

            message = choices[0].get("message", {})

            # 检查是否有 image 字段
            image_data = message.get("image")
            if image_data and isinstance(image_data, list) and image_data:
                collected_image_data = image_data[0].get("data") or image_data[0]

            # 收集 content 内容
            content_data = message.get("content")
            if content_data:
                collected_content = content_data

    # 优先返回 image 字段中的 base64 数据
    if collected_image_data:
        logger.info("使用 image 字段中的 base64 数据")
        return f"data:image/png;base64,{collected_image_data}"

    # 回退到从 content 中提取图片信息
    if collected_content:
        logger.info("从 content 中提取图片信息")
        # 提取 markdown 图片链接
        match = re.search(r"!\[.*?\]\((.*?)\)", collected_content)
        if match:
            url = match.group(1)
            logger.debug(f"从 markdown 提取图片 URL: {url}")
            return url

        # 如果 content 本身就是 URL
        if collected_content.startswith("http"):
            logger.debug(f"content 本身是图片 URL: {collected_content}")
            return collected_content.strip()
        
        # 如果 content 中包含图片数据
        if collected_content.startswith("data:image"):
            logger.debug("content 中包含 base64 图片数据")
            return collected_content.strip()
        
        # 或者，假设content中包含图片数据，但需要添加data头（存在数据？）
        if re.match(r"^[A-Za-z0-9+/=]+$", collected_content.strip()):
            logger.debug("content 中可能是纯 base64 图片数据，尝试添加 data 头")
            return f"data:image/png;base64,{collected_content.strip()}"

    # 都没有找到
    raise ValueError("未能从模型响应中提取图片，请检查模型是否支持图像生成或提示词是否合适")
