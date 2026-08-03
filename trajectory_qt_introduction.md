# 多模态足球机器人  
## 项目简介  
本项目是一款基于人工智能视觉识别技术与精密电机控制系统的智能足球守门员训练机器人，
旨在为职业足球俱乐部、青训、体校单招和守门员教学提供一套自动化、智能化的训练装备。
该项目的AI智能模式通过YOLO目标检测实时识别守门员站位与球门位置，
结合16分区网格瞄准算法和RS485通信协议精确控制发球机硬件，实现全自动智能对抗式训练。

### 一、核心创新点  
**1.AI视觉实时识别：** 基于自训练YOLO模型，同时检测守门员、球门横梁、左右立柱和足球5类目标，实现实时站位感知  
**2.16分区网格瞄准：** 将球门区域划分为4×4=16个网格训练区域，通过三轮转速+角度组合精准命中任意分区落点  
**3.EMA球门平滑跟踪：** 采用指数移动平均（EMA）算法，抑制抖动  
**4.三重安全保护：** 守门员必须位于球门区域内、不能离发球机过近、守门员回中时暂停发球  
**5.AI自适应发球策略：** 支持跟随发球、反向发球、随机发球三种AI智能模式，根据守门员站位动态调整发球落点  

### 二、技术栈  
**核心逻辑语言：** Python  
**计算机视觉与图像处理：**OpenCV+Numpy   
**目标检测：** YOLOv8n  
**平滑算法：** EMA指数移动平均  
**视频流：** RTSP协议实时摄像头  
**Web框架：** FastAPI+Uvicorn  
**数据校验：** Pydantic BaseModel   
**视频编码：** OpenCV VideoWriter  
**多线程：** Python threading+QThread

### 三、应用场景  
**职业、足球俱乐部：** 守门员日常专项扑救训练，替代人工喂球，提升训练效率与标准化程度  
**体育单招：** 提高足球体育单招的规范性和公平性  
**青训学院/足球学校：** 青少年守门员基础扑球定型训练，通过AI跟随模式保证训练连续性  
**专业教练培训：** 教练可通过手动模式自由调参，设计个性化训练方案，适配不同水平球员  
**赛前热身：** 比赛赛前快速热身扑救，AI反向发球模式模拟比赛射门场景  
**康复训练：** 术后恢复期的守门员渐进式训练，通过速度和落点精确控制训练强度  

### 四、系统整体架构  
┌─────────────────┐    ┌─────────────────┐     ┌─────────────────┐ 
│    视觉感知层     │────│    决策控制层     │──── │    硬件执行层     │
│   YOLO目标检测   │    │   16分区网格计算   │     │   RS485串口通信   │
│     安全检查     │    │    发球目标判定    │     │   三轮发球电机控制  │
│  EMA球门平滑跟踪  │    │    发球模式策略    │     │  左右/上下角度电机  │
│  守门员位置识别    │    │   FastAPI接口    │     └─────────────────┘
└─────────────────┘    └─────────────────┘  


### 五、项目目录结构  
football/ultralytics-main/   
├── datasets/                  # 训练所需的数据集目录   
│   └── football/              # 数据集  
│       ├── images/            # 图像数据    
│       └── labels/            # YOLO格式标签数据  
├── 1trajectory.py             # 批量视频处理+轨迹模拟（PyQt6）  
├── motor_simulator.py         # 电机操作模拟程序（RS485/JSON）  
├── runs/                      # 训练记录与检测结果输出路径  
│   └── detect/train           # 训练生成的目标检测模型权重存放处   
├── camera_test.py             # 摄像头模拟测试  
├── yolov8n.pt                 # 目标检测原始模型  
├── demo2/                     # 守门员准备样本  
└── football.yaml              # 训练目标检测模型的配置文件

### 六、核心实现  
#### EMA球门跟踪平滑  
def smooth_goal_box(prev, curr, alpha:0.25) -> GoalBox:  
    # 1.检查初始平滑值  
    if prev is None:
        return curr  
    # 2.平滑四条边界值  
    return GoalBox(  
        left   = prev.left   * (1.0 - alpha) + curr.left   * alpha,  
        top    = prev.top    * (1.0 - alpha) + curr.top    * alpha,  
        right  = prev.right  * (1.0 - alpha) + curr.right  * alpha,  
        bottom = prev.bottom * (1.0 - alpha) + curr.bottom * alpha,  
    )  

#### 球门组装算法  
def assemble_goal(dets, conf_thresh:0.3):  
    # 1.初始化横梁、左柱、右柱  
    crossbar = post_left = post_right = None  
    # 2.自动纠正左右柱顺序
    if crossbar and post_left and post_right:  
        if post_left.cx > post_right.cx:  
            post_left, post_right = post_right, post_left  
    # 3.计算最小矩形  
    left = min(p.x1 for p in parts)  
    top = min(p.y1 for p in parts)  
    right = max(p.x2 for p in parts)  
    bottom = max(p.y2 for p in parts)  
    # 4.返回四个边界坐标  
    return GoalBox(left, top, right, bottom)  

#### 分区网格系统  
def find_gk_zone(gk_det, goal) -> Tuple[int, int, int]:  
    # 1.生成4*4分区  
    xs = [goal.left + i * goal.width / GOAL_GRID_COLS for i in range(GOAL_GRID_COLS + 1)]  
    ys = [goal.top + i * goal.height / GOAL_GRID_ROWS for i in range(GOAL_GRID_ROWS + 1)]  
    # 2.分区编号  
    return row * GOAL_GRID_COLS + col + 1  
    # 3.计算守门员中心点所在分区    
    for c in range(GOAL_GRID_COLS):  
        if xs[c] <= gk_det.cx < xs[c + 1]:  
            col = c  
            break  

#### 安全检查模块  
def check_safety(gk_det, goal) -> Tuple[bool, str]:
    # 1.未检测到守门员  
    if gk_det is None:  
        return False, "未检测到守门员"  
    # 2.未检测到球门  
    if goal is None:
        return False, "未检测到球门"  
    # 3.检查守门员是否在球门框内  
    gk_box = (gk_det.x1, gk_det.y1, gk_det.x2, gk_det.y2)  
    goal_box = (goal.left, goal.top, goal.right, goal.bottom)  
    overlap_x1 = max(gk_box[0], goal_box[0])  
    overlap_y1 = max(gk_box[1], goal_box[1])  
    overlap_x2 = min(gk_box[2], goal_box[2])  
    overlap_y2 = min(gk_box[3], goal_box[3])  
    if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:  
        return False, "守门员不在球门区域内" 

#### 发球目标判定  
def determine_serve_target(mode, gk_side, preset_direction, target_row) -> Tuple[int, int]:  
    # 1.随机发球    
    if mode == "随机发球":  
        z = random.choice(ALL_ZONES)  
        return (z - 1) % 4, z  
    # 2.跟随发球和反向发球  
    if mode == "跟随发球":  
        target_col = near_col  
    elif mode == "反向发球":  
        target_col = far_col  

#### 球门可视化  
def draw_goal_zones(frame, goal, target_col, target_row, target_zone,
                    show_zone_numbers) -> np.ndarray:  
    # 1.绘制球门区域  
    overlay = frame.copy()  
    cv2.rectangle(overlay,  
                  (int(goal.left), int(goal.top)),  
                  (int(goal.right), int(goal.bottom)),  
                  (0, 220, 220), -1)  
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)  
    # 2.目标区域  
    if target_col >= 0 and target_row >= 0:  
        cv2.rectangle(tgt_overlay, (tx1, ty1), (tx2, ty2), (0, 0, 255), -1)  
        cv2.addWeighted(tgt_overlay, 0.35, frame, 0.65, 0, frame)  
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 0, 255), 4)  
    # 3.分区编号  
    if show_zone_numbers:
    for r in range(GOAL_GRID_ROWS):  
        for c in range(GOAL_GRID_COLS):  
            cx = int((xs[c] + xs[c + 1]) / 2)  
            cy = int((ys[r] + ys[r + 1]) / 2)  
            cv2.putText(frame, str(zone_number(r, c)), (cx - 8, cy + 5),  
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)  

### 七、核心数据结构
#### 球门检测框
@dataclass
class Detection:
    cls: int           # 类别ID (0=goalkeeper, 1=crossbar, 2=post_left, 3=post_right, 4=ball)    
    conf: float        # 置信度    
    x1: float          # 检测框左上角X  
    y1: float          # 检测框左上角Y  
    x2: float          # 检测框右下角X  
    y2: float          # 检测框右下角Y  
    @property  
    def cx(self): return (self.x1 + self.x2) / 2    # 中心X  
    @property  
    def cy(self): return (self.y1 + self.y2) / 2    # 中心Y  
    @property  
    def width(self): return self.x2 - self.x1       # 球门宽  
    @property  
    def height(self): return self.y2 - self.y1      # 球门高  

#### 球检测框
@dataclass  
class GoalBox:  
    left: float  
    top: float  
    right: float  
    bottom: float  
    @property  
    def width(self): return self.right - self.left  
    @property  
    def height(self): return self.bottom - self.top

#### 硬件状态参数   
hardware_state = {
    "serve_count": 0,            # 累计发球数  
    "speed": 50,                 # 当前速度  
    "difficulty": "normal",      # 难度等级  
    "serve_interval": 1.0,       # 发球间隔(秒)  
    "battery_level": 100,        # 电池电量  
    "motor_temp": 35.0,          # 电机温度  
    "is_safe": False,            # 安全状态  
    "safety_reason": "未启动",    # 安全原因  
}

#### 通信协议数据  
#1.PC -> 设备: 请求数据   
{    
    "PID": 1425,  
    "REQ": ["HARVR", "SOFVR", "STATE", "SBCNT", "BSTAT", "RUNTM"],  
    "CKS": 0  
}  
 
#2.设备 -> PC: 应答数据  
{
    "AID": 1425,  
    "RST": 0,            # 0=成功, 1~99=错误  
    "RES": {"HARVR": "SSB2301_V2", "SOFVR": "XG01_01", "STATE": "work"},  
    "CKS": 0  
}  
 
#3.PC -> 设备: 发球指令  
{  
    "PID": 10226,  
    "MDF": {  
        "SDATA": [50, 50, 50, 40, 50],   # 当前发球参数  
        "SNEXT": [50, 50, 50, 30, 30]     # 预备下一球参数  
    },  
    "CKS": 0  
}  
 
#4.设备 -> PC: 发球完成上报  
{  
    "PID": 27,  
    "RPT": {"SBCNT": 88, "SCMPL": 10226},  
    "CKS": 0  
}

接口路径	        方法	    功能说明	                                标签  
/start	        POST	启动机器人: 启动相机线程+算法线程+解锁电机	    核心控制  
/pause	        POST	暂停机器人: 锁定电机+暂停算法线程	            核心控制  
/stop	        POST	停止并复位: 停止线程+复位球门/GK状态+归零+锁定	核心控制  
/update_params	POST	参数配置: 更新速度/难度/发球间隔	            参数配置  
/current_data	GET	    数据查询: 获取整体状态(状态/速度/电量/安全等)	数据查询  
/serve_count	GET	    数据查询: 获取累计发球数量	                数据查询  
/video_feed	    GET	    视频流: 实时MJPEG画面推送	                视频流  

### 八、技术指标
#### 视觉识别性能指标
指标名称	        参数值	    说明  
检测模型	        YOLO    基于Ultralytics框架  
检测类别数	    5 类	守门员/横梁/左柱/右柱/足球  
守门员检测阈值	    0.20	置信度 >= 0.20 方可作为有效检测  
球门组件检测阈值	0.25	横梁/立柱置信度 >= 0.25  
足球检测阈值	    0.15	足球置信度 >= 0.15  
EMA平滑系数	    0.25	新值权重25%，历史值权重75%  
检测跟踪间隔	    2 帧	每2帧运行一次YOLO检测，平衡精度与性能  
球门最小尺寸	   30 像素	检测框宽高均须 >= 30px 方为有效球门  
安全重叠阈值	    15%	    守门员框至少15%在球门区域内方为安全  

#### 硬件性能指标  
指标名称	    参数值	         说明  
发球轮速范围	20 ~ 100	三轮独立可调，通过转速差产生多种球路  
左右角度范围	0 ~ 60	    水平方向发球角度调节  
上下角度范围	0 ~ 60	    垂直方向发球角度调节  
车轮速度范围	-100 ~ 100	100 = 1m/s，负值表示后退  
硬件版本	    SSB2301_V2	设备主控板硬件版本号  
软件版本	    XG01_01	    设备固件软件版本号  



