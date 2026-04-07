# 文件名：test_sim.py（环境探测器）
# 功能：验证仿真器是否能正常启动，不涉及 AI，纯粹检查硬件渲染。
# 先跑 test_sim.py：盯着屏幕，即使它黑屏也别动，等它弹出机械臂乱动的窗口。这说明“着色器编译”和“渲染引擎”通了。
# 再跑 main_sim.py：观察终端，模型加载会很慢。加载完后，它会根据你写在那行代码里的 instruction 开始尝试抓取。
import robosuite as suite  # 导入机器人仿真库
import numpy as np         # 导入数学库处理动作数组

# 1. 初始化环境
env = suite.make(
    env_name="PickPlace",  # 任务：抓取物体并投放
    robots="Panda",        # 机器人：使用经典的 Franka Panda 机械臂
    has_renderer=True,     # 开启图形化渲染界面
    use_camera_obs=False,  # 测试阶段不需要获取相机画面
)

print("仿真环境已启动，正在尝试随机动作...")
env.reset()  # 初始化/重置仿真世界

for i in range(100):
    # 2. 随机生成一个动作
    # OSC_POSE 控制器下，动作是一个 7 维向量：
    # [dx, dy, dz] - 三轴位移
    # [droll, dpitch, dyaw] - 三轴旋转角度
    # [gripper] - 夹爪开启/闭合 (通常 -1 到 1)
    action = np.random.uniform(-1, 1, env.action_dim)

    
    # 3. 执行动作
    # step 函数返回：新观测值、奖励值、是否结束、辅助信息
    obs, reward, done, info = env.step(action)
    
    # 4. 刷新渲染窗口
    env.render()

env.close() # 关闭环境
print("测试成功！")
