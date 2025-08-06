import os
import torch


class AnsiColor:
    # ANSI 码表
    _MAP = {
        # 基础颜色
        "black": 30, "red": 31, "green": 32, "yellow": 33,
        "blue": 34, "magenta": 35, "cyan": 36, "white": 37,
        # 亮颜色
        "bright_black": 90, "bright_red": 91, "bright_green": 92,
        "bright_yellow": 93, "bright_blue": 94, "bright_magenta": 95,
        "bright_cyan": 96, "bright_white": 97,
        # 样式
        "bold": 1, "dim": 2, "italic": 3, "underline": 4,
        "blink": 5, "reverse": 7, "strike": 9
    }

    _RESET = "\033[0m"
    _enabled = not os.getenv("NO_COLOR")  # 遵循 NO_COLOR 规范

    @classmethod
    def disable(cls):
        """全局关闭颜色"""
        cls._enabled = False

    @classmethod
    def enable(cls):
        """重新开启颜色"""
        cls._enabled = True

    def __call__(self, text: str, *color_or_style: str) -> str:
        """
        把 text 包装成带有颜色/样式的字符串。
        :param text: 原始字符串
        :param color_or_style: 任意数量的颜色名或样式名，例如 "red", "bold"
        :return: 带转义序列的新字符串
        """
        if not self._enabled or not color_or_style:
            return str(text)

        codes = [f"\033[{self._MAP[c]}m"
                 for c in color_or_style
                 if c in self._MAP]
        prefix = "".join(codes)
        return f"{prefix}{text}{self._RESET}"

color = AnsiColor()

def printTensor(t: torch.Tensor):
    shape = t.shape
    if len(shape) == 1:
        for i in range(shape[0]):
            print(f"{t[i]:.4f}", end=" ")
        print()
    elif len(shape) == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                print(f"{t[i, j]:.4f}", end=" ")
            print()
    else:
        t_for_print = t.view(-1, shape[-2], shape[-1])
        for i in range(t_for_print.shape[0]):
            print(f"Tensor {i}:")
            for j in range(t_for_print.shape[1]):
                for k in range(t_for_print.shape[2]):
                    print(f"{t_for_print[i, j, k]:.4f}", end=" ")
                print()
            print()
def printNamedTensor(name: str, t: torch.Tensor):
    print("=" * 50)
    print(color(f"{name}:", "blue", "bold"))
    printTensor(t)

def check(check_result: bool, message: str):
    if not check_result:
        print(color(f"Check Failed: {message}", "red", "bold"))
    else:
        print(color(f"Check Passed: {message}", "green", "bold"))

