from pathlib import Path
from typing import Union

from playwright.async_api import Page as AsyncPage
from playwright.sync_api import Page as SyncPage

# Mouse visualizer script (inline to avoid file dependencies)
MOUSE_VISUALIZER_SCRIPT = """
() => {
    // 如果已经存在跟踪器，则不重复创建
    if (document.getElementById('mouse-tracker')) {
        return;
    }

    // 创建鼠标跟踪器元素
    const tracker = document.createElement('div');
    tracker.id = 'mouse-tracker';

    // 设置跟踪器样式
    Object.assign(tracker.style, {
        position: 'fixed',
        width: '20px',
        height: '20px',
        borderRadius: '50%',
        backgroundColor: 'rgba(255, 0, 0, 0.5)',
        border: '2px solid red',
        pointerEvents: 'none',
        zIndex: '2147483647',  // 最大 z-index 值，确保在最上层
        transform: 'translate(-50%, -50%)',
        boxShadow: '0 0 10px rgba(255, 0, 0, 0.8)',
        transition: 'transform 0.05s ease-out'
    });

    // 创建坐标显示元素
    const coordinates = document.createElement('div');
    coordinates.id = 'mouse-coordinates';

    // 设置坐标显示样式
    Object.assign(coordinates.style, {
        position: 'fixed',
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        color: 'white',
        padding: '4px 8px',
        borderRadius: '4px',
        fontSize: '12px',
        fontFamily: 'monospace',
        pointerEvents: 'none',
        zIndex: '2147483647',
        transform: 'translate(10px, 10px)'
    });

    // 将元素添加到 body
    document.body.appendChild(tracker);
    document.body.appendChild(coordinates);

    // 添加鼠标移动事件监听器到 window 对象，确保全局捕获
    window.addEventListener('mousemove', (e) => {
        // 更新跟踪器位置
        tracker.style.left = `${e.clientX}px`;
        tracker.style.top = `${e.clientY}px`;

        // 更新坐标显示
        coordinates.textContent = `X: ${e.clientX}, Y: ${e.clientY}`;
        coordinates.style.left = `${e.clientX}px`;
        coordinates.style.top = `${e.clientY}px`;
    });

    // 添加鼠标点击效果
    window.addEventListener('mousedown', () => {
        tracker.style.backgroundColor = 'rgba(0, 255, 0, 0.5)';
        tracker.style.border = '2px solid green';
        tracker.style.transform = 'translate(-50%, -50%) scale(1.5)';
    });

    window.addEventListener('mouseup', () => {
        tracker.style.backgroundColor = 'rgba(255, 0, 0, 0.5)';
        tracker.style.border = '2px solid red';
        tracker.style.transform = 'translate(-50%, -50%) scale(1)';
    });

    console.log('Mouse tracker injected successfully');
}
"""


async def inject_mouse_visualizer_global_async(page: AsyncPage):
    """
    Inject mouse position visualization asynchronously in the Playwright page.

    Args:
        page: Playwright Asynchronous Page Object
    """
    await page.evaluate(expression=MOUSE_VISUALIZER_SCRIPT)


def inject_mouse_visualizer_global_sync(page: SyncPage):
    """
    Synchronously inject mouse position visualizations in the Playwright page.

    Args:
        page: Playwright Synchronize Page Objects
    """
    page.evaluate(MOUSE_VISUALIZER_SCRIPT)


async def inject_mouse_visualizer_global(page: Union[SyncPage, AsyncPage]):
    """
    Inject mouse position visualizations into the Playwright page, supporting synchronous and asynchronous APIs.

    Args:
        page: Playwright Page object, can be synchronous or asynchronous
    """
    if isinstance(page, AsyncPage):
        await inject_mouse_visualizer_global_async(page)
    else:
        inject_mouse_visualizer_global_sync(page)
