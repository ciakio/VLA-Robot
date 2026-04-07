import numpy as np
import robosuite as suite
from vla_model import OpenVLAgent
import cv2

def run_vla_grasping():
    print("初始化 Robosuite 环境 (指令压力测试模式)...")
    camera_name = "robot0_eye_in_hand" 
    
    env = suite.make(  
       env_name="PickPlace", 
       robots="Panda",             
       has_renderer=True,          
       has_offscreen_renderer=True, 
       use_camera_obs=True,        
       camera_names=camera_name,    
       controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
       horizon=2000 
    )
      
    agent = OpenVLAgent()
    obs = env.reset()

    # --- 任务调度器配置 ---
    # 定义三个极简阶段，测试模型对方向词的敏感度
    test_stages = [
        {"desc": "左移测试", "cmd": "move the arm to the left", "steps": 5},
        {"desc": "右移测试", "cmd": "move the arm to the right", "steps": 5},
        {"desc": "上移测试", "cmd": "move the arm up", "steps": 5}
    ]

    print("开始控制循环。准备执行‘左-右-上’三位一体自检...")
    
    try:
        for stage in test_stages:
            print(f"\n>>> 正在进行: {stage['desc']} | 指令: {stage['cmd']}")
            
            for s in range(stage['steps']):
                # 获取手眼图像
                img = obs[f"{camera_name}_image"]
                img = np.flipud(img)          
                img_uint8 = (img * 255).astype(np.uint8) 
                img_uint8 = cv2.resize(img_uint8, (224, 224), interpolation=cv2.INTER_AREA)

                # 显示画面
                display_img = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                cv2.putText(display_img, f"Stage: {stage['desc']}", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("VLA Monitor", display_img)

                # AI 决策
                action = agent.predict_action(img_uint8, stage['cmd'])

                # 打印原始参数辅助观察
                print(f"  [{stage['desc']} {s+1}/{stage['steps']}] AI位移: {np.round(action[:3], 4)}")

                # 动作处理
                if env.action_dim == 8 and len(action) == 7:
                    action = np.append(action, 0.0)
                
                action[-1] = 2.0 * (action[-1] - 0.5) # 夹爪
                action[:3] *= 2.5 # 适中放大

                # 执行 (Chunking 10 帧，模拟移动过程)
                for _ in range(10):
                    obs, reward, done, info = env.step(action)
                    env.render()
                    if done: break
                
                if cv2.waitKey(1) & 0xFF == ord('q'): return

        print("\n方向自检完成！如果机械臂确实做了‘左右上’的动作，说明‘人话’听懂了。")
        print("现在你可以放心地去写‘pick up the red block’这种带颜色的复杂指令了。")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vla_grasping()
