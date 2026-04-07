# 文件名：main_sim.py（系统联调提速优化版）
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
       controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
       # 【新增】：增加任务步数上限，从默认的 500 提高到 2000，给慢速 AI 留出足够时间
       horizon=2000 
    )
      
    # 2. 实例化 AI 大脑 
    agent = OpenVLAgent()

    # 3. 初始化场景
    obs = env.reset()
    # 提示：OpenVLA-7B 这种原子模型对多步复杂指令理解力有限，建议先从单步指令测试
    instruction = "Pick up each of the four blocks from the left tray and place them in the four corners of the work area one by one."

    print("开始控制循环。按 'q' 退出。")
    try:
        step_count = 0
        while True:
            # --- 步骤 A：图像预处理 ---
            img = obs["agentview_image"]
            img = np.flipud(img)          
            img_uint8 = (img * 255).astype(np.uint8) 
            img_uint8 = cv2.resize(img_uint8, (224, 224), interpolation=cv2.INTER_AREA)

            # --- 步骤 B：监视器预览 ---
            display_img = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            cv2.imshow("VLA Monitor (Press Q to quit)", display_img)

            # --- 步骤 C：AI 决策 (最耗时的一步) ---

            action = agent.predict_action(img_uint8, instruction)
            raw_xyz = action[:3]
            raw_gripper = action[-1]
            print(f"步数: {step_count:03d} | AI 原始位移: {np.round(raw_xyz, 4)} | 夹爪值: {round(raw_gripper, 2)}")
            # --- 步骤 D：动作对齐与转换 ---
            if env.action_dim == 8 and len(action) == 7:
                action = np.append(action, 0.0)
            
            action[-1] = 2.0 * (action[-1] - 0.5) # 夹爪映射
            action[:3] *= 1.5                    # 放大位移量

            # --- 步骤 E：物理模拟步进 (引入 Action Chunking 提速) ---
            # 【核心优化】：AI 思考一次，机械臂连续执行 8 帧动作。
            # 这利用了动作的连续性，能让运行速度提升 8 倍，且动作更连贯。
            for i in range(8):
                obs, reward, done, info = env.step(action)
                env.render()
                
                # 如果在重复执行期间任务结束了，直接跳出
                if done:
                    break

            # 【关键修复】：处理 Episode 结束逻辑，防止报错
            if done:
                print("检测到当前 Episode 结束，正在重置环境并开始新的一局...")
                obs = env.reset()
                continue # 跳回循环开始
            step_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vla_grasping()
