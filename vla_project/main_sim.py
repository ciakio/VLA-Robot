# 文件名：main_sim.py（系统联调核心）
# 功能：连接图像输入、AI 决策和机械臂执行的完整闭环。
# 输入自然语言 instruction = ""
# 可视化窗口1：OpenCV 窗口显示机器人相机拍到的画面
# 可视化窗口2：MuJoCo 3D 仿真窗口显示机械臂在仿真里真实运动的 3D 画面。
# 微调：如果机械臂动得太慢或太快，可以给 action[:3] 乘一个系数
# 运行前输入python -m bitsandbytes，以检测显卡驱动和 CUDA Toolkit 版本是否和bitsandbytes匹配
import numpy as np
import robosuite as suite
from vla_model import OpenVLAgent
import cv2

def run_vla_grasping():
    # 1. 深度配置环境
    print("初始化 Robosuite 环境...")
    env = suite.make(  
       env_name="PickPlace", 
       robots="Panda",             
       has_renderer=True,          
       has_offscreen_renderer=True, 
       use_camera_obs=True,        
       camera_names="agentview",    
       controller_configs=suite.load_controller_config(default_controller="OSC_POSE")
    )

    # 2. 实例化 AI 大脑 
    agent = OpenVLAgent()

    # 3. 初始化场景
    obs = env.reset()
    instruction = "pick up the red block and place it in the green bowl"

    print("开始控制循环。按 'q' 退出。")
    try:
        while True:
            # --- 步骤 A：图像预处理 ---
            # Robosuite 出来的图是 [0,1] 的浮点数，且坐标轴是反的
            img = obs["agentview_image"]
            img = np.flipud(img)          # 垂直翻转，让图片方向变正
            img_uint8 = (img * 255).astype(np.uint8) # 转为 0-255 的整数给 AI
            img_uint8 = cv2.resize(img_uint8, (224, 224), interpolation=cv2.INTER_AREA)

            # --- 步骤 B：监视器预览 ---
            display_img = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            cv2.imshow("VLA Monitor (Press Q to quit)", display_img)

            # --- 步骤 C：AI 决策 ---
            # 把图像和指令塞给 OpenVLA，拿到 7 维动作
            action = agent.predict_action(img_uint8, instruction)

            if env.action_dim == 8 and len(action) == 7:
                # 在末尾补一个 0.0，使长度变为 8
                action = np.append(action, 0.0)
            # --- 步骤 D：动作映射转换 ---
            # OpenVLA 的夹爪输出：0(闭合) 到 1(开启)
            # Robosuite 的夹爪输入：-1(闭合) 到 1(开启)
            vla_gripper = action[-1] 
            env_gripper = 2.0 * (vla_gripper - 0.5) 
            action[-1] = env_gripper

            # 微调：如果机械臂动得太慢，可以给 action[:3] 乘一个系数
            action[:3] *= 1.5  # 放大位移量

            # --- 步骤 E：物理模拟步进 ---
            obs, reward, done, info = env.step(action)

            # --- 步骤 F：实时渲染预览窗口 ---
            env.render()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vla_grasping()
