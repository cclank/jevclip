"""Make the synthetic test video behind the README's numbers.

Twelve blocks of Chinese blogger-style narration, each labelled keep or drop
in advance, and a video where every block is one solid colour, so the
highlight reel can be checked frame by frame. The narration is written to
exercise the rubric; it is not real speech and says nothing about accuracy
on real videos.

    python examples/make_demo.py [folder]      # default ./demo
    jevclip run demo/                          # needs TYPESAFE_API_KEY
    python examples/score_demo.py demo jevclip-out
"""
import json
import os
import re
import subprocess
import sys

BLOCKS = [
 ("drop", "寒暄", "哈喽大家好，欢迎回到我的频道，我是老K。好久不见了，最近一直在忙搬家的事情，家里乱糟糟的，镜头后面全是箱子，大家就当没看见哈。先跟大家说一声抱歉，上周答应的视频拖更了。也谢谢弹幕里一直催更的朋友，你们的留言我基本都看了。好，那我们闲话少说，今天这期来聊一个最近特别火的话题，评论区问得最多的就是它，今天一次性给大家讲清楚，大家坐稳了。"),
 ("drop", "夸大", "先说结论：星河七B绝对是目前全网最强的本地模型，没有之一，吊打所有闭源大模型。我可以负责任地告诉你，用了它，你以后再也不需要花钱买任何AI会员了，百分之百能帮你每年省下好几千块。我身边用过的朋友没有一个说不好的，内部消息说大厂都在偷偷用它。你要是现在还没用上，那真的已经落后百分之九十九的人了，这波不上车真的会后悔一辈子。"),
 ("keep", "实测数据", "我们来看具体测试。我用一台M5、32GB内存的MacBook Air，跑的是4-bit量化版本，模型文件4.3GB。同一段2000字的中文长文总结，首字延迟0.8秒，生成速度稳定在每秒38个token，整段跑完用了21秒，峰值内存6.1GB。作为对比，同样的任务用13B的上一代模型，速度只有每秒17个token，而且中途内存占到11GB。所以在这台机器上，7B这个版本速度大概是上一代的两倍多，内存差不多只占一半。"),
 ("drop", "广告", "在继续之前，插播一条广告。本期视频由某某云赞助，新用户注册就送一百小时GPU时长，链接我放在评论区置顶了。另外我自己的AI实战课也在打折，前五百名报名立减两百元，想系统学习的朋友千万别错过。还有，如果你觉得这期视频有帮助，麻烦一键三连，点赞投币收藏，再点个关注，你们的支持真的是我更新最大的动力，谢谢大家，我们继续。"),
 ("keep", "步骤", "下面说怎么在本地装起来，一共三步。第一步，安装Ollama，官网下载安装包，装好以后在终端输入ollama --version，能看到版本号就说明成功了。第二步，拉取模型，输入ollama pull加上模型名，冒号后面写q4_K_M，这样拉的就是4-bit量化版，大概4GB。第三步，想让它读长文档，要把上下文长度调大，在Modelfile里加一行PARAMETER num_ctx 16384，再用ollama create生成一个新模型，不然默认只有2048，长文章后半段会被直接截掉。"),
 ("drop", "空话", "其实吧，AI这个东西，怎么说呢，它就是一个工具，对吧。工具嘛，你用得好它就好，你用不好它就不好。所以说关键还是在于人，在于你怎么去用它。你说它重要吗，它肯定重要，但是你说它是不是万能的，那也不是万能的。所以我们要辩证地去看待这个问题，既不能太看高它，也不能太看低它，大概就是这么一个意思吧，这个大家自己去体会一下，每个人的理解可能都不一样。"),
 ("keep", "讲解", "很多人问量化到底是什么，这里简单讲一下原理。模型的权重本来是用16位浮点数存的，量化就是把每个权重压缩成更少的位数，比如4位。7B模型有70亿个参数，16位的时候大概要14GB，压到4位就只要3.5GB左右，再加上一些额外开销，就是我们看到的4GB多一点。代价是精度会有损失，但像q4_K_M这种方法会对更重要的层保留更高的精度，所以实际用起来大部分任务差别很小，这也是它能在笔记本上跑起来的原因。"),
 ("drop", "鸡汤", "说到这里我想多说两句。学习AI最重要的是什么？是坚持。很多人三天打鱼两天晒网，今天学一点明天就放下了，那肯定学不会。你看那些成功的人，哪个不是每天都在坚持学习？所以大家一定要保持好奇心，多动手多实践，不要怕犯错。机会永远是留给有准备的人的，只要你愿意付出努力，就一定会有收获。相信我，你一定可以的，我们一起加油，一起进步，一起在AI时代找到属于自己的位置。"),
 ("keep", "判断", "那本地模型到底适合谁？我的判断是，如果你的主要需求是处理隐私数据，比如公司合同、病历这类不能上传的东西，本地7B已经够用，值得折腾。但如果你追求的是最好的推理和写作质量，本地模型还是明显不如云端的旗舰模型，我在十道多步推理题上测过，它答对了四道，云端旗舰答对了九道。所以我的建议是分场景用，敏感数据走本地，复杂任务走云端，而不是非此即彼。"),
 ("drop", "闲聊", "说个题外话，前两天我去参加了一个线下的AI聚会，现场人特别多，大概来了两三百人吧。我在那碰到了好几个以前只在网上聊过的朋友，大家一起吃了个饭，聊得挺开心的。散场的时候都快十一点了，外面还下着雨，我打车等了半个多小时才回到家，到家都十二点多了，第二天差点起不来。反正那天挺累的，但是挺值的，下次有机会我再跟大家分享一下聚会上的一些照片。"),
 ("keep", "纠错", "还有一个常见误区要纠正一下。网上很多人说，内存越大模型就跑得越快，这个说法不对。我实际测了，同一个7B量化模型，在16GB和32GB的机器上，生成速度几乎一样，都是每秒37到38个token，因为模型文件只有4GB，两台机器都放得下。内存大小决定的是你能不能装下更大的模型，而速度主要取决于内存带宽和芯片的算力。所以如果你只跑7B，没必要为了速度去加内存。"),
 ("drop", "结尾", "好了，以上就是今天的全部内容。如果你喜欢这期视频，别忘了点赞关注，我们下期再见。对了，下周我可能会做一期更长的对比评测，大家可以在评论区告诉我想看哪些模型，拜拜。"),
]
RATE, PAUSE = 4.2, 1.2
COLORS = ["ff0000","00ff00","0000ff","ffff00","ff00ff","00ffff","ff8000","8000ff","008040","ffffff","804000","404040"]

def pieces(text):
    parts = [p for p in re.split(r"(?<=[，。！？；：])", text) if p]
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > 24:
            out.append(cur); cur = p
        else:
            cur += p
    if cur: out.append(cur)
    return out

def ts(x):
    ms = int(round(x * 1000))
    return "%02d:%02d:%02d,%03d" % (ms // 3600000, ms // 60000 % 60, ms // 1000 % 60, ms % 1000)

t, cues, spans = 0.0, [], []
for i, (label, name, text) in enumerate(BLOCKS):
    start = t
    # each block lasts 47-57 s (3.3-4.3 chars/s) so it lands in one 45 s segment
    total = len(text) / RATE if i == len(BLOCKS) - 1 else min(max(len(text) / RATE, 47.0), 57.0)
    for p in pieces(text):
        d = total * len(p) / len(text)
        cues.append((t, t + d, p)); t += d
    spans.append({"block": i + 1, "name": name, "label": label, "start": round(start, 2), "end": round(t, 2),
                  "chars": len(text), "seconds": round(t - start, 1), "color": COLORS[i]})
    if i < len(BLOCKS) - 1:
        t += PAUSE

bad = [s for s in spans[:-1] if not 45 <= s["seconds"] <= 60]
for s in spans: print("%2d %-4s %-4s %3d 字 %5.1f s  %s–%s" % (s["block"], s["label"], s["name"], s["chars"], s["seconds"], s["start"], s["end"]))
if bad: sys.exit("blocks outside 45–60 s: %s" % [b["block"] for b in bad])

folder = sys.argv[1] if len(sys.argv) > 1 else "demo"
os.makedirs(folder, exist_ok=True)
name = os.path.join(folder, "星河-7B本地部署实测")
with open(name + ".srt", "w", encoding="utf-8") as fh:
    fh.write("\n".join("%d\n%s --> %s\n%s\n" % (i, ts(a), ts(b), x) for i, (a, b, x) in enumerate(cues, 1)))
with open(os.path.join(folder, "labels.json"), "w", encoding="utf-8") as fh:
    json.dump(spans, fh, ensure_ascii=False, indent=1)

args, labels = [], []
for i, s in enumerate(spans):
    dur = (spans[i + 1]["start"] - s["start"]) if i + 1 < len(spans) else (s["end"] - s["start"])
    args += ["-f", "lavfi", "-i", "color=c=0x%s:s=320x180:r=25:d=%.3f" % (s["color"], dur)]
    labels.append("[%d:v]" % i)
n = len(spans)
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args,
                "-f", "lavfi", "-i", "sine=frequency=330:duration=%.3f" % t,
                "-filter_complex", "%sconcat=n=%d:v=1:a=0[v]" % ("".join(labels), n),
                "-map", "[v]", "-map", "%d:a" % n, "-c:v", "libx264", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", name + ".mp4"], check=True)
print("cues %d, total %.1f s -> %s.mp4 + .srt" % (len(cues), t, name))
