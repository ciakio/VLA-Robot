# 文件名：main_sim.py (感知与思维链优化版)
import numpy as np
import robosuite as suite
from vla_model import OpenVLAgent
import cv2
import torch

def run_vla_grasping():
    print("初始化 Robosuite 环境 (修正相机名称)...")
    
    # --- 关键修正：根据你的报错信息，改为 robot0_eye_in_hand ---
    camera_name = "robot0_eye_in_hand" 
    
    env = suite.make(  
       env_name="PickPlace", 
       robots="Panda",             
       has_renderer=True,          
       has_offscreen_renderer=True, 
       use_camera_obs=True,        
       camera_names=camera_name, # 这里修正了
       controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
       horizon=2000 
    )
      
    agent = OpenVLAgent()
    obs = env.reset()

    # --- 优化 B：Chain-of-Thought Prompting (思维链启发) ---
    # 不再只是简单的 goal，而是描述动作的逻辑过程
    instruction = (
        "Focus on the object in front of the gripper. "
        "Carefully move the arm down towards the object, "
        "then close the gripper firmly to pick it up and lift it."
    )

    print(f"当前思维链指令: {instruction}")
    
    try:
        step_count = 0
        while True:
            # --- 步骤 A：手眼图像预处理 ---
            # 注意：key 值需要对应 camera_name
            img = obs[f"{camera_name}_image"]
            img = np.flipud(img)          
            img_uint8 = (img * 255).astype(np.uint8) 
            img_uint8 = cv2.resize(img_uint8, (224, 224), interpolation=cv2.INTER_AREA)

            # 监视器实时画面（现在你会看到画面随着手部移动）
            display_img = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            cv2.putText(display_img, "Hand-Eye View", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("VLA Monitor", display_img)

            # --- 步骤 B：AI 决策 ---
            action = agent.predict_action(img_uint8, instruction)

            # --- 核心调试打印 ---
            raw_xyz = action[:3]
            raw_gripper = action[-1]
            print(f"步数: {step_count:03d} | AI原始位移: {np.round(raw_xyz, 4)} | 夹爪原始值: {round(raw_gripper, 2)}")

            # --- 步骤 C：动作对齐与参数平衡 ---
            if env.action_dim == 8 and len(action) == 7:
                action = np.append(action, 0.0)
            
            # 夹爪映射
            action[-1] = 2.0 * (action[-1] - 0.5) 
            
            # 【参数平衡】：减小单步幅度，增加灵敏度
            # 2.0 * 10 步 = 20倍放大，比之前的 60倍放大更适合近距离精准抓取
            action[:3] *= 2.0  

            # --- 步骤 D：物理执行 (Action Chunking) ---
            for _ in range(10):
                obs, reward, done, info = env.step(action)
                env.render()
                if done: break

            if done:
                print("检测到 Episode 结束，重置环境...")
                obs = env.reset()
                step_count = 0
                continue

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
