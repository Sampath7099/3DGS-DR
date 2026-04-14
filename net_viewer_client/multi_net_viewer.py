import os
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
import time, struct, socket

class MultiNetwork:
    def __init__(self, ports):
        self.conns = []
        for p in ports:
            try:
                c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                c.connect(("127.0.0.1", p))
                self.conns.append(c)
                print(f"Connected to port {p}")
            except Exception as e:
                print(f"Failed to connect to port {p}: {e}")
                self.conns.append(None)

    def read(self, idx):
        c = self.conns[idx]
        if c is None: return None
        try:
            len_buf = c.recv(32)
            if not len_buf: 
                self.conns[idx] = None
                return None
            length = int.from_bytes(len_buf, 'little')
            buf = b''
            while length:
                newbuf = c.recv(length)
                if not newbuf: break
                buf += newbuf
                length -= len(newbuf)
            return buf
        except:
            self.conns[idx] = None
            return None

    def send(self, idx, byte_data):
        c = self.conns[idx]
        if c is None: return
        try:
            data_len = len(byte_data).to_bytes(32, 'little')
            c.sendall(data_len)
            c.sendall(byte_data)
        except:
            self.conns[idx] = None
            print(f"Server {idx} disconnected.")

class OrbitCamera:
    def __init__(self, img_wh, center, r, rot = None):
        self.W, self.H = img_wh
        self.radius = r
        self.center = center
        self.rot = np.eye(3) if rot is None else rot

    @property
    def pose(self):
        res = np.eye(4)
        res[2, 3] -= self.radius
        rot = np.eye(4)
        rot[:3, :3] = self.rot.T
        res = rot @ res
        res[:3, 3] -= self.center
        return res

    def orbit(self, dx, dy):
        rotvec_x = self.rot[:, 1] * np.radians(0.05 * dx)
        rotvec_y = self.rot[:, 0] * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_y).as_matrix() @ \
                   R.from_rotvec(rotvec_x).as_matrix() @ \
                   self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 1e-4 * self.rot.T @ np.array([dx, dy, dz])

class MultiGUI:
    def __init__(self, ports):
        self.net = MultiNetwork(ports)
        self.num_views = len(ports)
        
        # Set absolute safe defaults in case NO servers are connected
        self.W, self.H = 800, 800
        self.center = np.zeros(3)
        self.radius = 1.0
        
        # Determine master dims from first active connection
        # and ensure WE DRAIN THE METADATA from all successful connections!
        master_found = False
        for i in range(self.num_views):
            if self.net.conns[i] is not None:
                try:
                    # Every connected server sends EXACTLY two messages at start
                    header_wh = self.net.read(i)
                    header_center = self.net.read(i)
                    
                    if header_wh and not master_found:
                        img_wh = struct.unpack('ii', header_wh)
                        self.W, self.H = img_wh[0], img_wh[1]
                        c_data = struct.unpack('ffff', header_center)
                        self.center = np.array([c_data[0], c_data[1], c_data[2]])
                        self.radius = c_data[3]
                        master_found = True
                        print(f"Server {i} initialized metadata: {self.W}x{self.H}")
                except Exception as e:
                    print(f"Error reading header from server {i}: {e}")

        self.cam = OrbitCamera((self.W, self.H), self.center, self.radius)
        self.render_buffers = [np.ones((self.H, self.W, 3), dtype=np.float32) for _ in range(self.num_views)]
        self.img_mode = 0
        self.dt = 0
        self.register_dpg()

    def register_dpg(self):
        dpg.create_context()
        # Viewport for 2 models side-by-side
        grid_w, grid_h = self.W * 2, self.H
        dpg.create_viewport(title="Dual Model Viewer", width=grid_w, height=grid_h)

        with dpg.texture_registry(show=False):
            for i in range(self.num_views):
                dpg.add_raw_texture(self.W, self.H, self.render_buffers[i], 
                                   format=dpg.mvFormat_Float_rgb, tag=f"_texture_{i}")

        with dpg.window(tag="_primary_window", width=grid_w, height=grid_h):
            with dpg.group(horizontal=True):
                for i in range(self.num_views):
                    dpg.add_image(f"_texture_{i}")
        
        dpg.set_primary_window("_primary_window", True)

        def callback_mode_select(sender, app_data):
            modes = {'color':0, 'strength':1, 'base_col':2, 'refl_col':3, 'normal':4, 'cluster':5}
            self.img_mode = modes[app_data]

        with dpg.window(label="Sync Controls", width=200, height=150, pos=[0, 0]):
            dpg.add_radio_button(items=['color', 'strength', 'base_col', 'refl_col', 'normal', 'cluster'], 
                                   callback=callback_mode_select)
            dpg.add_text('Syncing 4 views', tag="_log_time")

        def callback_drag(sender, app_data):
            if dpg.is_item_focused("_primary_window"): self.cam.orbit(app_data[1], app_data[2])
        def callback_wheel(sender, app_data):
            if dpg.is_item_focused("_primary_window"): self.cam.scale(app_data)
        def callback_pan(sender, app_data):
            if dpg.is_item_focused("_primary_window"): self.cam.pan(app_data[1], app_data[2])

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_drag)
            dpg.add_mouse_wheel_handler(callback=callback_wheel)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_pan)

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def render_loop(self):
        while dpg.is_dearpygui_running():
            t1 = time.time()
            pose_bytes = self.cam.pose.astype('float32').flatten().tobytes()
            mode_bytes = struct.pack('i', self.img_mode)
            
            # Send to all
            for i in range(self.num_views):
                self.net.send(i, mode_bytes)
                self.net.send(i, pose_bytes)
            
            # Read from all
            for i in range(self.num_views):
                ret = self.net.read(i)
                if ret:
                    rgb = np.frombuffer(ret, dtype=np.float32).reshape(self.H, self.W, 3)
                    dpg.set_value(f"_texture_{i}", rgb)
            
            self.dt = time.time() - t1
            dpg.set_value("_log_time", f'Frame time: {1000*self.dt:.2f} ms')
            dpg.render_dearpygui_frame()

if __name__ == "__main__":
    ports = [12357, 12358]
    gui = MultiGUI(ports)
    gui.render_loop()
    dpg.destroy_context()
