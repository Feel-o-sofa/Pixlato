import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImagePalette
from PIL.Image import DecompressionBombError
import os
import sys
import json
import threading
import time

# Import core logic
from core.common import TaskManager, OperationCancelled, debug_log
from core.processor import (EngineDispatcher, pixelate_image, upscale_for_preview, add_outline, 
                            remove_background, apply_grain_effect, remove_background_ai, 
                            remove_background_interactive, is_directml_supported,
                            normalize_image_geometry)
from core.palette import apply_palette_unified
from core.project_manager import ProjectManager
from core.gif_processor import process_gif
from core.image_manager import ImageManager
from ui.components import IntSpinbox, CustomPaletteWindow, ToolTip, PaletteInspector, BatchExportWindow, PluginWindow, bind_ctk_slider_wheel
from ui.theme_manager import ThemeManager
from ui.locale_manager import LocaleManager
from core.plugin_engine import PluginEngine

class PixelApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # 1. Start initialization sequence
        self._init_managers()
        self._init_state()
        
        # 2. Window Setup
        self.title("Pixlato - Pixel Art Studio")
        self.geometry("1400x1000")
        self.resizable(True, True)
        self.minsize(1000, 700)

        # Set App Icon
        try:
            if sys.platform.startswith("win") and os.path.exists(self.ico_path):
                self.iconbitmap(self.ico_path)
            elif os.path.exists(self.logo_path):
                icon_img = Image.open(self.logo_path)
                self.iconphoto(False, ImageTk.PhotoImage(icon_img))
        except Exception as e:
            print(f"Failed to set app icon: {e}")

    def _init_managers(self):
        """Initializes all singleton managers and engines."""
        current_dir = os.path.dirname(os.path.abspath(__file__))  
        self.project_root = os.path.dirname(current_dir)
        self.assets_dir = os.path.join(self.project_root, "assets")
        self.logo_path = os.path.join(self.assets_dir, "logo.png")
        self.ico_path = os.path.join(self.assets_dir, "logo.ico")

        self.locale = LocaleManager(self.assets_dir, default_lang="ko")
        self.theme_manager = ThemeManager()
        self.image_manager = ImageManager()
        self.plugin_engine = PluginEngine(os.path.join(self.project_root, "plugins"))
        self.plugin_engine.discover_plugins()
        self.dml_supported = is_directml_supported()

    def _init_state(self):
        """Initializes application state variables."""
        self.palette_dir = "palettes"
        if not os.path.exists(self.palette_dir): os.makedirs(self.palette_dir)
        self.last_palette_dir = self.palette_dir
        self.presets_path = os.path.join(self.project_root, "palettes", "presets.json")
        
        self.original_image_path = None
        self.original_size = (0, 0)
        self.raw_pixel_image = None   
        self.preview_image = None     
        self.mag_zoom = 3
        self.user_palette_colors_persistent = [(0,0,0)] * 16 
        self.active_palette_mode = "Standard" 
        self.preview_zoom = 1.0 
        self.current_inventory_id = None
        self.inventory_widgets = {}
        self.inventory_reset_buttons = {}
        self.inventory_active_checkboxes = {}
        self.presets = {}
        self.bg_seeds = []
        self.fg_seeds = []
        self._is_processing = False
        self._pending_reprocess = False
        self._is_restoring = False
        
        # Dependency on methods that need UI defaults
        self.global_params = self.get_hardcoded_defaults()
        self.cached_defaults = self.get_hardcoded_defaults()
        self.last_processed_params = {} # Cache for change detection

        self.load_default_palette()
        self.load_presets()
        
        # 2. Window Setup (Continued)
        self._create_navbar()
        self._create_sidebar()
        self._create_canvas_area()
        self._create_inventory_area()
        
        # 3. Apply Layout
        self._do_main_layout()
        self._regrid_sidebar()

        # 4. Finalize initialization
        self._finalize_init()

    def _do_main_layout(self):
        """Standardizes the overall application grid layout."""
        # Main Window Grid
        self.grid_columnconfigure(0, weight=0, minsize=320)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=180)
        self.grid_rowconfigure(1, weight=1)

        # 1. Navbar Layout
        self.navbar.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.nav_label.pack(side="left", padx=20)
        self.btn_load_pal.pack(side="left", padx=5)
        self.btn_save_pal.pack(side="left", padx=5)
        self.proj_label.pack(side="left", padx=(30, 10))
        self.btn_load_proj.pack(side="left", padx=5)
        self.btn_save_proj.pack(side="left", padx=5)
        self.option_lang.pack(side="right", padx=20)

        # 2. Sidebar Skeleton
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        
        self.logo_label.pack(pady=(20, 10), padx=20)
        self.label_setting_mode.pack(pady=(5, 0), padx=20, fill="x")
        self.setting_mode_switch.pack(pady=5, padx=20, fill="x")
        self.label_engine_mode.pack(pady=(10, 0), padx=20, fill="x")
        self.engine_switch.pack(pady=5, padx=20, fill="x")

        # Bottom Area
        self.status_container.pack(side="bottom", pady=0, fill="x")
        self.status_container.pack_propagate(False)
        self.status_label.pack(expand=True)
        self.bottom_action_frame.pack(side="bottom", fill="x", padx=10, pady=10)
        self.btn_open_mag.pack(pady=5, fill="x")
        self.btn_plugin_mgr.pack(pady=5, fill="x")
        self.btn_save.pack(pady=5, fill="x")

        # Scroll Sidebar - Remove outer padding to stretch to edges
        self.scroll_sidebar.pack(side="top", fill="both", expand=True)
        self.label_mode.pack(pady=(15, 0), padx=20, fill="x")
        self.mode_switch.pack(pady=(5, 10), padx=20, fill="x")
        
        # Ensure the scrollable frame's internal container stretches
        self.param_frame.pack(fill="x", expand=True)
        self.param_frame.grid_columnconfigure(0, weight=1)

        # 3. Canvas Area Layout
        self.preview_frame.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")
        self.preview_frame.grid_propagate(False)
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        
        self.palette_inspector.place(relx=0.0, rely=0.0, x=15, y=15, anchor="nw")
        self.label_palette_sort.place(relx=0.0, rely=0.0, x=315, y=2, anchor="nw")
        self.sort_menu.place(relx=0.0, rely=0.0, x=325, y=17, anchor="nw")
        self.btn_export_gpl.place(relx=0.0, rely=0.0, x=420, y=17, anchor="nw")

        self.res_info_frame.place(relx=1.0, rely=0.0, x=-15, y=15, anchor="ne")
        self.res_info_frame.pack_propagate(False)
        self.res_label.pack(expand=True)

        # 4. Inventory Layout
        self.inventory_frame.grid(row=1, column=2, sticky="nsew")
        self.inventory_frame.grid_propagate(False)
        self.btn_add_to_inv.pack(pady=10, padx=10, fill="x")
        self.label_inv_header.pack(pady=(5, 0))
        self.inventory_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        self.btn_batch_export.pack(pady=10, padx=10, fill="x", side="bottom")

    def _regrid_sidebar(self):
        """Orchestrates visibility of sidebar elements without flickering (using grid/grid_remove)."""
        pal_mode = self.option_palette.get()
        is_auto = self.check_auto_optimal.get()
        extract_pol = self._get_logical(self.extract_policy_switch.get(), "extract_policy")
        bg_mode = self._get_logical(self.option_bg_mode.get(), "bg_mode")

        # 1. Palette Group internal visibility
        if pal_mode in ["Limited", "Grayscale"]:
            # Auto Optimal Checkbox
            self.check_auto_optimal.pack(pady=5, padx=20, fill="x")
            
            # Color Limit Slider Group
            if not is_auto:
                self.color_limit_group.grid() # Re-show with remembered options
            else:
                self.color_limit_group.grid_remove() # Hide but keep metadata
        else:
            self.check_auto_optimal.pack_forget()
            self.color_limit_group.grid_remove()

        # 2. RAP Aesthetics (Conditional visibility)
        if pal_mode == "Limited" and extract_pol == "Aesthetic":
            self.rap_frame.grid()
        else:
            self.rap_frame.grid_remove()

        # 3. Background removal helpers
        if bg_mode == "Interactive":
            self.btn_clear_seeds.pack(pady=2, padx=20, fill="x")
            self.label_interactive_helper.pack(pady=2, padx=20, fill="x")
        else:
            self.btn_clear_seeds.pack_forget()
            self.label_interactive_helper.pack_forget()

        # 4. FX Expanded items (Edge & Grain)
        if self.check_edge_enhance.get():
            self.edge_controls_container.pack(fill="x", padx=(35, 20), pady=(0, 5))
        else:
            self.edge_controls_container.pack_forget()

        if self.check_grain.get():
            self.grain_controls_container.pack(fill="x", padx=(35, 20), pady=(0, 5))
        else:
            self.grain_controls_container.pack_forget()

    def _create_navbar(self):
        """Creates the top navigation bar and its children."""
        self.navbar = ctk.CTkFrame(self, height=40, corner_radius=0)
        
        self.nav_label = ctk.CTkLabel(self.navbar, text="", font=("Arial", 12, "bold"))
        self.locale.register(self.nav_label, "nav_palette_mgmt")

        self.btn_load_pal = ctk.CTkButton(self.navbar, text="", width=80, height=28, command=self.load_palette_file)
        self.locale.register(self.btn_load_pal, "nav_load")
        self.theme_manager.register_widget(self.btn_load_pal)

        self.btn_save_pal = ctk.CTkButton(self.navbar, text="", width=120, height=28, command=self.save_palette_file)
        self.locale.register(self.btn_save_pal, "nav_save_as")
        self.theme_manager.register_widget(self.btn_save_pal)

        self.proj_label = ctk.CTkLabel(self.navbar, text="", font=("Arial", 12, "bold"))
        self.locale.register(self.proj_label, "nav_project")

        self.btn_load_proj = ctk.CTkButton(self.navbar, text="", width=120, height=28, command=self.load_project_file)
        self.locale.register(self.btn_load_proj, "nav_open_pcp")
        self.theme_manager.register_widget(self.btn_load_proj)

        self.btn_save_proj = ctk.CTkButton(self.navbar, text="", width=100, height=28, command=self.save_project_file)
        self.locale.register(self.btn_save_proj, "nav_save_pcp")
        self.theme_manager.register_widget(self.btn_save_proj)

        self.option_lang = ctk.CTkOptionMenu(self.navbar, width=80, height=28, values=self.locale.get_available_languages(), command=self.change_language)
        self.option_lang.set(self.locale.current_lang)
        self.theme_manager.register_widget(self.option_lang)

    def _create_sidebar(self):
        """Creates the sidebar and all its internal control groups."""
        # Main sidebar container
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color="#1a1a1a")
        # No grid placement here, handled by _do_main_layout
        
        self.scroll_sidebar = ctk.CTkScrollableFrame(self.sidebar, label_text="CONTROL PANEL", 
                                                   label_font=("Arial", 12, "bold"),
                                                   corner_radius=0, fg_color="transparent")
        
        # Access internal widgets of CTkScrollableFrame to eliminate gaps
        try:
            self.scroll_sidebar._parent_canvas.configure(highlightthickness=0)
            self.scroll_sidebar._content_frame.configure(padx=0, pady=0)
        except: pass
        
        self.param_frame = self.scroll_sidebar
        self._create_sidebar_groups()

        # 2a. Top Sidebar Labels & Mode Switches
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Pixlato ✨", font=ctk.CTkFont(size=28, weight="bold"))

        self.label_setting_mode = ctk.CTkLabel(self.sidebar, text="", anchor="w")
        self.locale.register(self.label_setting_mode, "sidebar_setting_mode")

        self.setting_mode_switch = ctk.CTkSegmentedButton(self.sidebar, values=[self.locale.get("mode_global"), self.locale.get("mode_individual")], command=self.on_setting_mode_change)
        self.setting_mode_switch.set(self.locale.get("mode_individual"))
        self.theme_manager.register_widget(self.setting_mode_switch)

        self.label_engine_mode = ctk.CTkLabel(self.sidebar, text="", anchor="w")
        self.locale.register(self.label_engine_mode, "sidebar_engine_mode")

        self.engine_switch = ctk.CTkSegmentedButton(self.sidebar, 
                                                    values=[self.locale.get("engine_auto"), 
                                                            self.locale.get("engine_cpu"), 
                                                            self.locale.get("engine_gpu")], 
                                                    command=self.on_engine_change)
        self.engine_switch.set(self.locale.get("engine_auto"))
        self.theme_manager.register_widget(self.engine_switch)

        # 2b. Fixed Bottom Sidebar Area
        self.bottom_action_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.btn_open_mag = ctk.CTkButton(self.bottom_action_frame, text="", command=self.toggle_magnifier, fg_color="#d35400", hover_color="#e67e22")
        self.locale.register(self.btn_open_mag, "sidebar_open_mag")
        self.mag_window = None

        self.btn_plugin_mgr = ctk.CTkButton(self.bottom_action_frame, text="", command=self.open_plugin_manager, fg_color="#16a085", hover_color="#1abc9c")
        self.locale.register(self.btn_plugin_mgr, "sidebar_plugin_mgr", prefix="🔌 ")

        self.btn_save = ctk.CTkButton(self.bottom_action_frame, text="", command=self.save_image, state="disabled", height=45, font=("Arial", 16, "bold"))
        self.locale.register(self.btn_save, "sidebar_export")
        self.theme_manager.register_widget(self.btn_save, role="success")

        self.status_container = ctk.CTkFrame(self.sidebar, height=30, fg_color="transparent")
        self.status_label = ctk.CTkLabel(self.status_container, text="", text_color="#2ecc71")

        # 2c. Scrollable Params Area
        self.scroll_sidebar = ctk.CTkScrollableFrame(self.sidebar, width=300, corner_radius=0, fg_color="transparent")
        self._create_sidebar_groups()

    def _create_sidebar_groups(self):
        """Creates grouped UI controls for the scrollable sidebar."""
        # Policy Group
        self.label_mode = ctk.CTkLabel(self.scroll_sidebar, text="", anchor="w")
        self.locale.register(self.label_mode, "sidebar_save_mode")
        self.mode_switch = ctk.CTkSegmentedButton(self.scroll_sidebar, values=[self.locale.get("save_pixelate"), self.locale.get("save_style")], command=self.on_param_change)
        self.mode_switch.set(self.locale.get("save_pixelate"))
        self.theme_manager.register_widget(self.mode_switch)

        self.param_frame = ctk.CTkFrame(self.scroll_sidebar, fg_color="transparent")
        self.param_frame.pack(fill="x", expand=True) # Full width stretch
        self.param_frame.grid_columnconfigure(0, weight=1)

        # 1. Pixel Size Group
        self.pixel_group = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        self.pixel_group.grid(row=0, column=0, sticky="ew", pady=(10, 5))
        self.pixel_group.grid_columnconfigure(0, weight=1)
        
        pixel_header = ctk.CTkFrame(self.pixel_group, fg_color="transparent")
        pixel_header.pack(fill="x", padx=15)
        self.label_pixel = ctk.CTkLabel(pixel_header, text="", anchor="w", width=80)
        self.locale.register(self.label_pixel, "sidebar_pixel_size")
        self.pixel_spin = IntSpinbox(pixel_header, from_=1, to=128, width=100, command=self.update_pixel_from_spinbox)
        self.pixel_spin.set(2)
        self.label_pixel.pack(side="left")
        self.pixel_spin.pack(side="right")
        
        self.slider_pixel = ctk.CTkSlider(self.pixel_group, from_=1, to=128, number_of_steps=127, command=self.update_pixel_from_slider)
        self.slider_pixel.set(2)
        self.slider_pixel.pack(pady=(2, 5), padx=15, fill="x")
        self.theme_manager.register_widget(self.slider_pixel)
        bind_ctk_slider_wheel(self.slider_pixel)

        # 2. Palette & Optimization Group
        self.palette_group = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        self.palette_group.grid(row=1, column=0, sticky="ew", pady=5)
        self.palette_group.grid_columnconfigure(0, weight=1)
        
        self.label_pal_preset = ctk.CTkLabel(self.palette_group, text="", anchor="w")
        self.locale.register(self.label_pal_preset, "sidebar_palette_preset")
        self.label_pal_preset.pack(pady=(5, 0), padx=15, fill="x")
        
        palette_values = ["Limited", "Original", "Grayscale", "GameBoy", "CGA", "Pico-8", "16-bit (4096 Colors)", "USER CUSTOM"]
        self.option_palette = ctk.CTkOptionMenu(self.palette_group, values=palette_values, command=self.on_palette_menu_change)
        self.option_palette.set("Limited")
        self.option_palette.pack(pady=5, padx=15, fill="x")
        self.theme_manager.register_widget(self.option_palette)

        self.check_auto_optimal = ctk.CTkCheckBox(self.palette_group, text="", command=self.on_auto_optimal_toggle)
        self.locale.register(self.check_auto_optimal, "sidebar_auto_optimal")
        self.theme_manager.register_widget(self.check_auto_optimal)
        # Not packed yet, managed by _regrid_sidebar

        # 3. Color Limit Group (Conditional result, but fixed position)
        self.color_limit_group = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        self.color_limit_group.grid(row=2, column=0, sticky="ew", pady=5)
        self.color_limit_group.grid_columnconfigure(0, weight=1)
        
        color_header = ctk.CTkFrame(self.color_limit_group, fg_color="transparent")
        color_header.pack(fill="x", padx=15)
        self.label_color_count = ctk.CTkLabel(color_header, text="", anchor="w", width=80)
        self.locale.register(self.label_color_count, "sidebar_color_limit")
        self.color_spinbox = IntSpinbox(color_header, from_=2, to=256, width=100, command=self.update_color_from_spinbox)
        self.color_spinbox.set(16)
        self.label_color_count.pack(side="left")
        self.color_spinbox.pack(side="right")
        
        self.color_slider = ctk.CTkSlider(self.color_limit_group, from_=2, to=256, number_of_steps=254, command=self.update_color_from_slider)
        self.color_slider.set(16)
        self.color_slider.pack(pady=(2, 5), padx=15, fill="x")
        self.theme_manager.register_widget(self.color_slider)
        bind_ctk_slider_wheel(self.color_slider)

        # Policy switches inside Palette Group (Bottom part)
        self.label_extract_policy = ctk.CTkLabel(self.palette_group, text="", anchor="w", wraplength=280, justify="left")
        self.locale.register(self.label_extract_policy, "sidebar_extract_policy")
        self.extract_policy_switch = ctk.CTkSegmentedButton(self.palette_group, values=[self.locale.get("policy_standard"), self.locale.get("policy_aesthetic")], 
                                                           command=lambda v: self.on_palette_menu_change(self.option_palette.get()))
        self.extract_policy_switch.set(self.locale.get("policy_standard"))
        self.theme_manager.register_widget(self.extract_policy_switch)
        self.label_extract_policy.pack(pady=(10, 0), padx=15, fill="x")
        self.extract_policy_switch.pack(pady=5, padx=15, fill="x")

        self.label_mapping_policy = ctk.CTkLabel(self.palette_group, text="", anchor="w", wraplength=280, justify="left")
        self.locale.register(self.label_mapping_policy, "sidebar_mapping_policy")
        self.mapping_policy_switch = ctk.CTkSegmentedButton(self.palette_group, values=[self.locale.get("policy_classic"), self.locale.get("policy_perceptual")], command=self.on_param_change)
        self.mapping_policy_switch.set(self.locale.get("policy_classic"))
        self.theme_manager.register_widget(self.mapping_policy_switch)
        
        self.label_mapping_policy.pack(pady=(10, 0), padx=15, fill="x")
        self.mapping_policy_switch.pack(pady=5, padx=15, fill="x")

        self.btn_custom_pal = ctk.CTkButton(self.palette_group, text="", command=self.open_custom_palette, fg_color="#8e44ad", hover_color="#9b59b6")
        self.locale.register(self.btn_custom_pal, "sidebar_edit_custom_pal")
        self.btn_custom_pal.pack(pady=5, padx=15, fill="x")
        
        self.check_dither = ctk.CTkCheckBox(self.palette_group, text="", command=self.on_param_change)
        self.locale.register(self.check_dither, "sidebar_dithering")
        self.theme_manager.register_widget(self.check_dither)
        self.check_dither.pack(pady=5, padx=15, fill="x")

        # 4. RAP Weights Group
        self.rap_frame = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        self.rap_frame.grid(row=4, column=0, sticky="ew", pady=5)
        self.rap_frame.grid_columnconfigure(0, weight=1)
        self.label_rap_header = ctk.CTkLabel(self.rap_frame, text="", anchor="w", font=("Arial", 11, "bold"))
        self.locale.register(self.label_rap_header, "sidebar_rap_weights")
        self.label_rap_header.pack(pady=(5, 0), padx=15, fill="x")
        
        self.rap_weights = {}
        for key, lang_key in [("sat", "weight_sat"), ("con", "weight_con"), ("rar", "weight_rar")]:
            f = ctk.CTkFrame(self.rap_frame, fg_color="transparent")
            f.pack(fill="x", pady=2, padx=15)
            lbl = ctk.CTkLabel(f, text="", anchor="w", font=("Arial", 10))
            self.locale.register(lbl, lang_key)
            val_lbl = ctk.CTkLabel(f, text="0.4" if key=="sat" else "0.3", width=30, font=("Arial", 10))
            lbl.pack(side="left")
            val_lbl.pack(side="right")
            
            sld = ctk.CTkSlider(self.rap_frame, from_=0, to=1.0, number_of_steps=20, height=16,
                                command=lambda v, l=val_lbl: [l.configure(text=f"{v:.2f}"), self.on_param_change()])
            sld.set(0.4 if key=="sat" else 0.3)
            sld.pack(fill="x", padx=15, pady=(0, 5))
            self.theme_manager.register_widget(sld)
            bind_ctk_slider_wheel(sld)
            self.rap_weights[key] = (sld, val_lbl, f, lbl)

        # 5. Visual Effects Group
        self.fx_group = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        self.fx_group.grid(row=5, column=0, sticky="ew", pady=5)
        self.fx_group.grid_columnconfigure(0, weight=1)
        
        self.label_vis_fx = ctk.CTkLabel(self.fx_group, text="", anchor="w", font=("Arial", 12, "bold"))
        self.locale.register(self.label_vis_fx, "sidebar_visual_effects")
        self.label_vis_fx.pack(pady=(10, 0), padx=15, fill="x")

        # BG Mode
        self.bg_frame = ctk.CTkFrame(self.fx_group, fg_color="transparent")
        self.bg_frame.pack(fill="x", pady=2, padx=15)
        self.label_bg_mode = ctk.CTkLabel(self.bg_frame, text="", anchor="w", font=("Arial", 11))
        self.locale.register(self.label_bg_mode, "sidebar_bg_mode")
        self.label_bg_mode.pack(side="left")
        
        bg_options = [self.locale.get("bg_none"), self.locale.get("bg_classic")]
        if self.dml_supported: bg_options.append(self.locale.get("bg_ai"))
        bg_options.append(self.locale.get("bg_interactive"))

        self.option_bg_mode = ctk.CTkOptionMenu(self.fx_group, values=bg_options, command=self.on_bg_mode_change)
        self.option_bg_mode.set(self.locale.get("bg_none"))
        self.option_bg_mode.pack(pady=2, padx=15, fill="x")
        self.theme_manager.register_widget(self.option_bg_mode)

        self.btn_clear_seeds = ctk.CTkButton(self.fx_group, text="", height=24, fg_color="#7f8c8d", command=self.clear_bg_seeds)
        self.locale.register(self.btn_clear_seeds, "btn_clear_seeds")
        self.label_interactive_helper = ctk.CTkLabel(self.fx_group, text="", font=("Arial", 11), text_color="#e67e22", wraplength=280)

        self.check_outline = ctk.CTkCheckBox(self.fx_group, text="", command=self.on_param_change)
        self.locale.register(self.check_outline, "sidebar_outline")
        self.theme_manager.register_widget(self.check_outline)
        self.check_outline.pack(pady=5, padx=15, fill="x")

        # Edge Enhancement
        self.edge_group = ctk.CTkFrame(self.fx_group, fg_color="transparent")
        self.edge_group.pack(fill="x", pady=5)
        self.check_edge_enhance = ctk.CTkCheckBox(self.edge_group, text="", command=self.on_edge_enhance_toggle)
        self.locale.register(self.check_edge_enhance, "sidebar_edge_enhance")
        self.check_edge_enhance.pack(fill="x", padx=15)
        self.theme_manager.register_widget(self.check_edge_enhance)

        self.edge_controls_container = ctk.CTkFrame(self.edge_group, fg_color="transparent")
        edge_header = ctk.CTkFrame(self.edge_controls_container, fg_color="transparent")
        edge_header.pack(fill="x", pady=(2, 0))
        self.label_edge_sens_header = ctk.CTkLabel(edge_header, text="", anchor="w", font=("Arial", 10))
        self.locale.register(self.label_edge_sens_header, "sidebar_edge_sens")
        self.label_edge_sens = ctk.CTkLabel(edge_header, text="1.0", width=40, anchor="e", font=("Arial", 10))
        self.label_edge_sens_header.pack(side="left")
        self.label_edge_sens.pack(side="right")
        
        self.slider_edge_sens = ctk.CTkSlider(self.edge_controls_container, from_=0, to=10.0, number_of_steps=100, height=16, command=self.update_edge_sens_label)
        self.slider_edge_sens.set(1.0)
        self.slider_edge_sens.pack(fill="x", pady=(0, 5))
        self.theme_manager.register_widget(self.slider_edge_sens)

        # Grain Effect
        self.grain_group = ctk.CTkFrame(self.fx_group, fg_color="transparent")
        self.grain_group.pack(fill="x", pady=5)
        self.check_grain = ctk.CTkCheckBox(self.grain_group, text="", command=self.on_grain_toggle)
        self.locale.register(self.check_grain, "sidebar_grain")
        self.check_grain.pack(fill="x", padx=15)
        self.theme_manager.register_widget(self.check_grain)
        
        self.grain_controls_container = ctk.CTkFrame(self.grain_group, fg_color="transparent")
        grain_header = ctk.CTkFrame(self.grain_controls_container, fg_color="transparent")
        grain_header.pack(fill="x", pady=(2, 0))
        self.label_grain_amt = ctk.CTkLabel(grain_header, text="", anchor="w", font=("Arial", 10))
        self.locale.register(self.label_grain_amt, "sidebar_grain_amt")
        self.label_grain_val = ctk.CTkLabel(grain_header, text="15", width=40, anchor="e", font=("Arial", 10))
        self.label_grain_amt.pack(side="left")
        self.label_grain_val.pack(side="right")
        
        self.slider_grain = ctk.CTkSlider(self.grain_controls_container, from_=0, to=50, number_of_steps=50, height=16, command=self.update_grain_label)
        self.slider_grain.set(15)
        self.slider_grain.pack(fill="x", pady=(0, 5))
        self.theme_manager.register_widget(self.slider_grain)

        # 6. Global Setup & Themes
        self.setup_group = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        self.setup_group.grid(row=6, column=0, sticky="ew", pady=10)
        self.setup_group.grid_columnconfigure(0, weight=1)
        
        self.label_downsample = ctk.CTkLabel(self.setup_group, text="", anchor="w", wraplength=280, justify="left")
        self.locale.register(self.label_downsample, "sidebar_downsample")
        self.label_downsample.pack(pady=(5, 0), padx=15, fill="x")
        self.option_downsample = ctk.CTkOptionMenu(self.setup_group, values=[self.locale.get("opt_standard"), self.locale.get("opt_kmeans")], command=self.on_param_change)
        self.option_downsample.set(self.locale.get("opt_standard"))
        self.option_downsample.pack(pady=5, padx=15, fill="x")
        self.theme_manager.register_widget(self.option_downsample)

        self.label_presets_header = ctk.CTkLabel(self.setup_group, text="", anchor="w", font=("Arial", 12, "bold"))
        self.locale.register(self.label_presets_header, "sidebar_presets")
        self.label_presets_header.pack(pady=(15, 0), padx=15, fill="x")
        self.preset_frame_inner = ctk.CTkFrame(self.setup_group, fg_color="transparent")
        self.preset_frame_inner.pack(fill="x", pady=5)
        self.option_presets = ctk.CTkOptionMenu(self.preset_frame_inner, values=[self.locale.get("preset_default")], command=self.apply_preset)
        self.option_presets.set(self.locale.get("preset_default"))
        self.option_presets.pack(side="left", padx=(15, 5), fill="x", expand=True)
        self.theme_manager.register_widget(self.option_presets)
        self.btn_save_preset = ctk.CTkButton(self.preset_frame_inner, text="💾", width=30, command=self.save_preset_dialog)
        self.btn_save_preset.pack(side="right", padx=(0, 15))
        self.theme_manager.register_widget(self.btn_save_preset)

        self.label_theme_header = ctk.CTkLabel(self.setup_group, text="", anchor="w", font=("Arial", 12, "bold"))
        self.locale.register(self.label_theme_header, "sidebar_theme")
        self.label_theme_header.pack(pady=(5, 0), padx=15, fill="x")
        self.option_theme = ctk.CTkOptionMenu(self.setup_group, values=self.theme_manager.get_available_themes(), command=self.change_theme)
        self.option_theme.set("Default Dark")
        self.option_theme.pack(pady=5, padx=15, fill="x")
        self.btn_custom_accent = ctk.CTkButton(self.setup_group, text="", command=self.pick_custom_accent)
        self.locale.register(self.btn_custom_accent, "btn_accent_picker")
        self.btn_custom_accent.pack(pady=2, padx=15, fill="x")
        self.theme_manager.register_widget(self.option_theme)
        
        self.btn_custom_accent = ctk.CTkButton(self.setup_group, text="🎨 Accent Color", height=24, command=self.pick_custom_accent)
        self.theme_manager.register_widget(self.btn_custom_accent)

    def _create_canvas_area(self):
        """Creates the main preview area and its overlays."""
        self.preview_frame = ctk.CTkFrame(self, corner_radius=12, fg_color="#1a1a1a")
        self.preview_canvas = ctk.CTkCanvas(self.preview_frame, bg="#1a1a1a", highlightthickness=0)
        
        self.palette_inspector = PaletteInspector(self.preview_frame, width=300, height=28, click_callback=self.on_palette_color_click, corner_radius=6, border_width=1, border_color="#444")
        
        self.palette_sort_mode = "Original"
        self.label_palette_sort = ctk.CTkLabel(self.preview_frame, text="Sort:", font=("Arial", 10), text_color="#888")
        self.locale.register(self.label_palette_sort, "sort_palette")

        self.sort_menu = ctk.CTkOptionMenu(self.preview_frame, values=["Original", "Luminance", "Hue"], width=90, height=24, font=("Arial", 10), command=self.on_palette_sort_change)
        self.sort_menu.set("Original")
        self.theme_manager.register_widget(self.sort_menu)

        self.btn_export_gpl = ctk.CTkButton(self.preview_frame, text="⬇ GPL", width=50, height=24, font=("Arial", 10, "bold"), command=self.export_used_palette_gpl)
        self.theme_manager.register_widget(self.btn_export_gpl)
        ToolTip(self.btn_export_gpl, "GIMP Palette (.gpl) Export")

        self.res_info_frame = ctk.CTkFrame(self.preview_frame, width=120, height=28, corner_radius=6, border_width=1, border_color="#444", fg_color="#2b2b2b")
        self.res_label = ctk.CTkLabel(self.res_info_frame, text="0 x 0", font=("Consolas", 12, "bold"), text_color="#aaa")

    def _create_inventory_area(self):
        """Creates the right inventory sidebar."""
        self.inventory_frame = ctk.CTkFrame(self, width=180, corner_radius=0)
        self.btn_add_to_inv = ctk.CTkButton(self.inventory_frame, text="", command=self.add_image_to_inventory, height=35)
        self.locale.register(self.btn_add_to_inv, "inv_add")
        self.theme_manager.register_widget(self.btn_add_to_inv, role="success")

        self.label_inv_header = ctk.CTkLabel(self.inventory_frame, text="", font=("Arial", 11))
        self.inventory_scroll = ctk.CTkScrollableFrame(self.inventory_frame, width=160)
        
        self.btn_batch_export = ctk.CTkButton(self.inventory_frame, text="", command=self.batch_export, height=35, fg_color="#3498db")
        self.locale.register(self.btn_batch_export, "inv_save_all")

    def _finalize_init(self):
        """Standardizes event bindings and initial UI sync."""
        self.bind_all("<Escape>", lambda e: self.cancel_current_task())
        self.preview_canvas.bind("<MouseWheel>", self.on_preview_wheel)
        self.preview_canvas.bind("<Motion>", self.update_magnifier)
        self.preview_canvas.bind("<Configure>", self.on_resize)
        self.preview_canvas.bind("<Button-1>", self.on_canvas_click)
        self.preview_canvas.bind("<Button-3>", self.on_canvas_click)
        self.canvas_image_id = None
        self.tk_preview = None

        self.update_ui_text()
        self.default_params = self.get_hardcoded_defaults()
        self.update_edge_controls_state()
        self.update_grain_controls_state()

    def get_hardcoded_defaults(self):
        """Returns the absolute initial default parameters for the app."""
        return {
            "save_mode": "Pixelate",
            "pixel_size": 2,
            "color_count": 16,
            "palette_mode": "Limited",
            "extract_policy": "Standard",
            "mapping_policy": "Classic",
            "rap_w_sat": 0.4,
            "rap_w_con": 0.3,
            "rap_w_rar": 0.3,
            "bg_mode": "None",
            "bg_seeds": [],
            "fg_seeds": [],
            "dither": False,
            "outline": False,
            "edge_enhance": False,
            "edge_sensitivity": 1.0,
            "grain_enabled": False,
            "grain_intensity": 15,
            "auto_optimal": False,
            "downsample_method": "Standard",
            "custom_colors": [(0, 0, 0)] * 16
        }

    def is_params_modified(self, params):
        """Perform a deep comparison between current params and cached defaults."""
        if not params: return False
        
        # Keys that use float comparison with tolerance
        float_keys = ["rap_w_sat", "rap_w_con", "rap_w_rar", "edge_sensitivity"]
        
        # Regular keys
        keys_to_check = [
            "save_mode", "pixel_size", "color_count", "palette_mode",
            "extract_policy", "mapping_policy", "bg_mode", "dither", 
            "outline", "edge_enhance", "grain_enabled", "grain_intensity", 
            "downsample_method", "auto_optimal"
        ]
        
        for k in keys_to_check:
            if params.get(k) != self.cached_defaults.get(k):
                return True
                
        for k in float_keys:
            v1 = params.get(k, 0)
            v2 = self.cached_defaults.get(k, 0)
            if abs(float(v1) - float(v2)) > 1e-5:
                return True
        
        # Deep comparison for lists
        if list(params.get("bg_seeds", [])) != list(self.cached_defaults.get("bg_seeds", [])): return True
        if list(params.get("fg_seeds", [])) != list(self.cached_defaults.get("fg_seeds", [])): return True
        if [tuple(c) for c in params.get("custom_colors", [])] != [tuple(c) for c in self.cached_defaults.get("custom_colors", [])]: return True
        
        return False

    def update_reset_button_visibility(self, iid, force_check=False):
        """
        Updates the visibility of the reset button for a specific item.
        Optimized using 'is_dirty' bit to avoid redundant comparisons.
        """
        e = self.image_manager.get_image(iid)
        if not e or iid not in self.inventory_reset_buttons: return
        
        btn = self.inventory_reset_buttons[iid]
        
        # Optimization: If already dirty, skip comparison unless force_check is true
        if not force_check and e.get("is_dirty"):
            btn.place(relx=1.0, rely=0, anchor="ne", x=-32, y=5)
            return

        # Perform comparison and update dirty bit
        modified = self.is_params_modified(e.get("params"))
        e["is_dirty"] = modified
        
        if modified:
            btn.place(relx=1.0, rely=0, anchor="ne", x=-32, y=5)
        else:
            # Explicitly hide tooltip if it exists before hiding the button
            if hasattr(btn, "_tooltip_ref"):
                btn._tooltip_ref.leave()
            btn.place_forget()

    # --- Methods ---
    def load_presets(self):
        if os.path.exists(self.presets_path):
            try:
                with open(self.presets_path, "r", encoding="utf-8") as f:
                    self.presets = json.load(f)
                names = list(self.presets.keys())
                if names: self.option_presets.configure(values=[self.locale.get("preset_default")] + names)
            except Exception as e: print(f"Error loading presets: {e}")

    def apply_preset(self, preset_name):
        if preset_name in self.presets: self.restore_ui_state(self.presets[preset_name])

    def pick_custom_accent(self):
        from tkinter import colorchooser
        current = self.theme_manager.get_current_accent()
        color = colorchooser.askcolor(initialcolor=current, title="Pick Accent Color", parent=self)
        if color[1]:
            self.theme_manager.apply_custom_color("accent", color[1])
            # Update OptionMenu values to include "Custom"
            themes = self.theme_manager.get_available_themes()
            self.option_theme.configure(values=themes)
            self.option_theme.set("Custom")

    def change_theme(self, theme_name):
        self.theme_manager.set_theme(theme_name)

    def change_language(self, lang_code):
        if self.locale.load_language(lang_code): self.update_ui_text()

    def open_plugin_manager(self):
        PluginWindow(self, self.plugin_engine, on_change_callback=self.on_param_change)

    def update_ui_text(self):
        self.locale.refresh_widgets()
        self.update_inventory_count_label()
        
        # Update Segmented Buttons
        self.setting_mode_switch.configure(values=[self.locale.get("mode_global"), self.locale.get("mode_individual")])
        self.mode_switch.configure(values=[self.locale.get("save_pixelate"), self.locale.get("save_style")])
        self.extract_policy_switch.configure(values=[self.locale.get("policy_standard"), self.locale.get("policy_aesthetic")])
        self.mapping_policy_switch.configure(values=[self.locale.get("policy_classic"), self.locale.get("policy_perceptual")])
        
        # Update Option Menus
        self.option_downsample.configure(values=[self.locale.get("opt_standard"), self.locale.get("opt_kmeans")])
        
        bg_options = [self.locale.get("bg_none"), self.locale.get("bg_classic")]
        if self.dml_supported:
            bg_options.append(self.locale.get("bg_ai"))
        bg_options.append(self.locale.get("bg_interactive"))
        self.option_bg_mode.configure(values=bg_options)
        
        current_presets = list(self.presets.keys())
        self.option_presets.configure(values=[self.locale.get("preset_default")] + current_presets)
        if self.option_presets.get() in ["기본값 (Defaults)", "Defaults", "기본값"]:
            self.option_presets.set(self.locale.get("preset_default"))
            
        # Standardize layout after text changes
        self._regrid_sidebar()

    def load_default_palette(self):
        path = os.path.join(self.palette_dir, "default_palette.json")
        if os.path.exists(path): self._load_palette_from_path(path)

    def _load_palette_from_path(self, path):
        try:
            with open(path, "r") as f:
                colors = json.load(f)
                if isinstance(colors, list) and len(colors) > 0:
                    self.user_palette_colors_persistent[:len(colors)] = [tuple(c) for c in colors]
        except Exception as e: print(f"Error loading palette: {e}")

    def save_default_palette(self):
        path = os.path.join(self.palette_dir, "default_palette.json")
        try:
            with open(path, "w") as f: json.dump(self.user_palette_colors_persistent, f)
        except Exception as e: print(f"Error saving palette: {e}")

    def load_palette_file(self):
        f = filedialog.askopenfilename(parent=self, initialdir=self.last_palette_dir, filetypes=[("JSON files", "*.json")])
        if f: 
            self.last_palette_dir = os.path.dirname(f)
            self._load_palette_from_path(f)
            self.option_palette.set("USER CUSTOM")
            self.on_palette_menu_change("USER CUSTOM")

    def save_palette_file(self):
        f = filedialog.asksaveasfilename(parent=self, initialdir=self.last_palette_dir, defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if f: 
            self.last_palette_dir = os.path.dirname(f)
            self.save_default_palette()

    def save_project_file(self):
        if not self.original_image_path: return
        state = self.capture_ui_state()
        state["source_path"] = self.original_image_path
        state["preview_zoom"] = self.preview_zoom
        f = filedialog.asksaveasfilename(parent=self, defaultextension=".pcp", filetypes=[("Pixel Crafter Project", "*.pcp")])
        if f: ProjectManager.save_project(f, state)

    def load_project_file(self):
        f = filedialog.askopenfilename(parent=self, filetypes=[("Pixel Crafter Project", "*.pcp")])
        if not f: return
        state = ProjectManager.load_project(f)
        if state is None: messagebox.showerror("보안 오류", "파일 손상"); return
        try:
            if "source_path" in state and os.path.exists(state["source_path"]):
                self.original_image_path = state["source_path"]
                with Image.open(self.original_image_path) as img: self.original_size = img.size
                self.preview_zoom = state.get("preview_zoom", -1)
            self.restore_ui_state(state)
            self.btn_save.configure(state="normal")
        except Exception as e: print(f"Error project load: {e}")


    def on_palette_menu_change(self, value):
        """Handler for palette menu selection."""
        # Update logical state
        self.active_palette_mode = "Custom_User" if value == "USER CUSTOM" else ("Custom_16bit" if value == "16-bit (4096 Colors)" else "Standard")
        
        # Delegate UI update to orchestrator
        self._regrid_sidebar()
        self.on_param_change()

    def on_palette_color_click(self, index, color): self.open_custom_palette(initial_index=index)

    def open_custom_palette(self, initial_index=0):
        current = self.palette_inspector.colors if self.palette_inspector.colors else self.user_palette_colors_persistent
        CustomPaletteWindow(self, self.on_custom_palette_applied, initial_colors=current, initial_index=initial_index, live_callback=self.on_live_palette_update)

    def on_live_palette_update(self, colors):
        if colors: 
            self.user_palette_colors_persistent[:len(colors)] = colors
            self.active_palette_mode = "Custom_User"
            self.on_param_change()

    def on_custom_palette_applied(self, mode, colors):
        if colors: 
            self.user_palette_colors_persistent[:len(colors)] = colors
            self.save_default_palette()
        self.active_palette_mode = "Custom_User"
        self.option_palette.set("USER CUSTOM")
        self.on_palette_menu_change("USER CUSTOM")

    def update_pixel_from_slider(self, value): self.pixel_spin.set(int(value)); self.on_param_change()
    def update_pixel_from_spinbox(self): 
        v = self.pixel_spin.get()
        if v is not None: self.slider_pixel.set(v); self.on_param_change()

    def update_color_from_slider(self, value): self.color_spinbox.set(int(value)); self.on_param_change()
    def update_color_from_spinbox(self):
        v = self.color_spinbox.get()
        if v is not None: 
            v = max(2, min(256, v))
            self.color_spinbox.set(v)
            self.color_slider.set(v)
            self.on_param_change()

    def update_edge_sens_label(self, value): self.label_edge_sens.configure(text=f"{float(value):.1f}"); self.on_param_change()
    def on_edge_enhance_toggle(self):
        self.update_edge_controls_state()
        self._regrid_sidebar() # Real-time margin cleanup
        self.on_param_change()
    def update_edge_controls_state(self):
        if self.check_edge_enhance.get(): 
            self.slider_edge_sens.configure(state="normal", progress_color=self.theme_manager.get_current_accent())
            self.label_edge_sens.configure(text_color="#FFFFFF")
        else: 
            self.slider_edge_sens.configure(state="disabled", progress_color="gray")
            self.label_edge_sens.configure(text_color="gray")

    def update_grain_label(self, value): 
        self.label_grain_val.configure(text=f"{int(value)}")
        self.on_param_change()

    def on_grain_toggle(self):
        self.update_grain_controls_state()
        self._regrid_sidebar() # Real-time margin cleanup
        self.on_param_change()
    def update_grain_controls_state(self):
        if self.check_grain.get():
            self.slider_grain.configure(state="normal", progress_color=self.theme_manager.get_current_accent())
            self.label_grain_val.configure(text_color="#FFFFFF")
        else:
            self.slider_grain.configure(state="disabled", progress_color="gray")
            self.label_grain_val.configure(text_color="gray")

    def on_bg_mode_change(self, value):
        mode = self._get_logical(value, "bg_mode")
        if mode == "Interactive":
            self.status_label.configure(text=self.locale.get("msg_interactive_tip"), text_color="#e67e22")
            self.preview_canvas.configure(cursor="hand2")
        else:
            self.status_label.configure(text="", text_color="#2ecc71")
            self.preview_canvas.configure(cursor="")
        
        self._regrid_sidebar()
        self.on_param_change()

    def on_auto_optimal_toggle(self):
        """Handler for auto-optimal checkbox."""
        self._regrid_sidebar()
        self.on_param_change()

    def clear_bg_seeds(self):
        self.bg_seeds = []
        self.fg_seeds = []
        self.on_param_change()

    def on_canvas_click(self, event):
        mode = self._get_logical(self.option_bg_mode.get(), "bg_mode")
        if mode != "Interactive" or not self.preview_image: return
        
        # Map click to original image coordinates
        cw, ch = self.preview_canvas.winfo_width(), self.preview_canvas.winfo_height()
        iw, ih = self.preview_image.size
        nw, nh = int(iw * self.preview_zoom), int(ih * self.preview_zoom)
        
        rel_x = (event.x - (cw//2 - nw//2)) / self.preview_zoom
        rel_y = (event.y - (ch//2 - nh//2)) / self.preview_zoom
        
        if 0 <= rel_x < iw and 0 <= rel_y < ih:
            # Button-1 (Left) is BG, Button-3 (Right) is FG
            if event.num == 1:
                self.bg_seeds.append((rel_x, rel_y))
            else:
                self.fg_seeds.append((rel_x, rel_y))
            self.on_param_change()

    def set_ui_mode(self, mode):
        """
        Orchestrates transition between Global and Individual settings modes.
        Used as a Single Source of Truth for sync between Sidebar and Inventory.
        """
        valid_modes = ["Global", "Individual"]
        if mode not in valid_modes: return
        
        # 1. Logic Transition & State Capturing
        if mode == "Global":
            self.global_params = self.capture_ui_state()
            self.status_label.configure(text="🌐 Global Mode: Temporary state active")
            
            # Auto-selection logic for excluded items
            if self.current_inventory_id is not None:
                e = self.image_manager.get_image(self.current_inventory_id)
                if e and not e.get("is_active_global", True):
                    all_entries = self.image_manager.get_all()
                    next_active = next((item for item in all_entries if item.get("is_active_global", True)), None)
                    if next_active:
                        self.current_inventory_id = next_active["id"]
                    else:
                        self.current_inventory_id = None
        else:
            # Global -> Individual: Apply current global settings to active items (Commit)
            import copy
            all_entries = self.image_manager.get_all()
            for entry in all_entries:
                if entry.get("is_active_global", True):
                    entry["params"] = copy.deepcopy(self.global_params)
                    self.update_reset_button_visibility(entry["id"], force_check=True)
            
            self.status_label.configure(text="👤 Individual Mode: Global settings applied to group")

        # 2. Update physical switch state
        display_val = self._get_display(mode, "setting_mode")
        if self.setting_mode_switch.get() != display_val:
            self.setting_mode_switch.set(display_val)

        # 3. Synchronize UI Components
        self._regrid_sidebar()
        self.update_inventory_appearance()
        
        # 4. Trigger reprocessing based on new mode
        if self.current_inventory_id is not None:
            self.select_inventory_image(self.current_inventory_id)
        elif self.original_image_path:
            self.process_image()

    def on_setting_mode_change(self, value):
        mode = self._get_logical(value, "setting_mode")
        self.set_ui_mode(mode)

    def _get_logical(self, display_val, category):
        m = {
            "save_mode": {self.locale.get("save_pixelate"): "Pixelate", self.locale.get("save_style"): "Style Only"},
            "downsample": {self.locale.get("opt_standard"): "Standard", self.locale.get("opt_kmeans"): "K-Means"},
            "setting_mode": {self.locale.get("mode_global"): "Global", self.locale.get("mode_individual"): "Individual"},
            "extract_policy": {self.locale.get("policy_standard"): "Standard", self.locale.get("policy_aesthetic"): "Aesthetic"},
            "mapping_policy": {self.locale.get("policy_classic"): "Classic", self.locale.get("policy_perceptual"): "Perceptual"},
            "bg_mode": {
                self.locale.get("bg_none"): "None", 
                self.locale.get("bg_classic"): "Classic",
                self.locale.get("bg_ai"): "AI Auto",
                self.locale.get("bg_interactive"): "Interactive"
            }
        }
        return m.get(category, {}).get(display_val, display_val)

    def _get_display(self, val, cat):
        m = {
            "save_mode": {"Pixelate": self.locale.get("save_pixelate"), "Style Only": self.locale.get("save_style")},
            "downsample": {"Standard": self.locale.get("opt_standard"), "K-Means": self.locale.get("opt_kmeans")},
            "setting_mode": {"Global": self.locale.get("mode_global"), "Individual": self.locale.get("mode_individual")},
            "extract_policy": {"Standard": self.locale.get("policy_standard"), "Aesthetic": self.locale.get("policy_aesthetic")},
            "mapping_policy": {"Classic": self.locale.get("policy_classic"), "Perceptual": self.locale.get("policy_perceptual")},
            "bg_mode": {
                "None": self.locale.get("bg_none"),
                "Classic": self.locale.get("bg_classic"),
                "AI Auto": self.locale.get("bg_ai"),
                "Interactive": self.locale.get("bg_interactive")
            }
        }
        return m.get(cat, {}).get(val, val)

    def capture_ui_state(self):
        return {
            "save_mode": self._get_logical(self.mode_switch.get(), "save_mode"), 
            "pixel_size": int(self.pixel_spin.get()), 
            "color_count": int(self.color_spinbox.get()), 
            "palette_mode": self.option_palette.get(), 
            "extract_policy": self._get_logical(self.extract_policy_switch.get(), "extract_policy"),
            "mapping_policy": self._get_logical(self.mapping_policy_switch.get(), "mapping_policy"),
            "rap_w_sat": float(self.rap_weights["sat"][0].get()),
            "rap_w_con": float(self.rap_weights["con"][0].get()),
            "rap_w_rar": float(self.rap_weights["rar"][0].get()),
            "bg_mode": self._get_logical(self.option_bg_mode.get(), "bg_mode"),
            "bg_seeds": list(self.bg_seeds),
            "fg_seeds": list(self.fg_seeds),
            "dither": self.check_dither.get(), 
            "outline": self.check_outline.get(), 
            "edge_enhance": self.check_edge_enhance.get(), 
            "edge_sensitivity": float(self.slider_edge_sens.get()), 
            "grain_enabled": self.check_grain.get(),
            "grain_intensity": int(self.slider_grain.get()),
            "auto_optimal": self.check_auto_optimal.get(),
            "downsample_method": self._get_logical(self.option_downsample.get(), "downsample"), 
            "custom_colors": list(self.user_palette_colors_persistent)
        }

    def restore_ui_state(self, params):
        if not params: return
        self._is_restoring = True
        try:
            if "auto_optimal" in params:
                (self.check_auto_optimal.select() if params["auto_optimal"] else self.check_auto_optimal.deselect())
            
            if "save_mode" in params: self.mode_switch.set(self._get_display(params["save_mode"], "save_mode"))
            if "pixel_size" in params: 
                self.slider_pixel.set(params["pixel_size"])
                self.pixel_spin.set(params["pixel_size"])
            if "color_count" in params: 
                self.color_slider.set(params["color_count"])
                self.color_spinbox.set(params["color_count"])
            if "palette_mode" in params: 
                self.option_palette.set(params["palette_mode"])
                self.on_palette_menu_change(params["palette_mode"])
            if "extract_policy" in params: self.extract_policy_switch.set(self._get_display(params["extract_policy"], "extract_policy"))
            if "mapping_policy" in params: self.mapping_policy_switch.set(self._get_display(params["mapping_policy"], "mapping_policy"))

            if "bg_mode" in params:
                target_bg = params["bg_mode"]
                if target_bg == "AI Auto" and not self.dml_supported:
                    target_bg = "None"
                self.option_bg_mode.set(self._get_display(target_bg, "bg_mode"))
                self.on_bg_mode_change(self.option_bg_mode.get())
            if "bg_seeds" in params: self.bg_seeds = list(params["bg_seeds"])
            if "fg_seeds" in params: self.fg_seeds = list(params["fg_seeds"])

            if "rap_w_sat" in params:
                self.rap_weights["sat"][0].set(params["rap_w_sat"])
                self.rap_weights["sat"][1].configure(text=f"{params['rap_w_sat']:.2f}")   
            if "rap_w_con" in params:
                self.rap_weights["con"][0].set(params["rap_w_con"])
                self.rap_weights["con"][1].configure(text=f"{params['rap_w_con']:.2f}")   
            if "rap_w_rar" in params:
                self.rap_weights["rar"][0].set(params["rap_w_rar"])
                self.rap_weights["rar"][1].configure(text=f"{params['rap_w_rar']:.2f}")   

            if "dither" in params: (self.check_dither.select() if params["dither"] else self.check_dither.deselect())
            if "outline" in params: (self.check_outline.select() if params["outline"] else self.check_outline.deselect())
            if "edge_enhance" in params:
                (self.check_edge_enhance.select() if params["edge_enhance"] else self.check_edge_enhance.deselect())
                self.update_edge_controls_state()
            if "edge_sensitivity" in params:
                self.slider_edge_sens.set(params["edge_sensitivity"])
                self.label_edge_sens.configure(text=f"{params['edge_sensitivity']:.1f}")  
            if "grain_enabled" in params:
                (self.check_grain.select() if params["grain_enabled"] else self.check_grain.deselect())
                self.update_grain_controls_state()
            if "grain_intensity" in params:
                self.slider_grain.set(params["grain_intensity"])
                self.label_grain_val.configure(text=f"{int(params['grain_intensity'])}")  
            if "downsample_method" in params: self.option_downsample.set(self._get_display(params["downsample_method"], "downsample"))
            
            # Update custom colors (must update list in-place or replace fully)
            if "custom_colors" in params:
                new_colors = [tuple(c) for c in params["custom_colors"]]
                self.user_palette_colors_persistent[:len(new_colors)] = new_colors
            
            self._is_restoring = False
            self.on_param_change()
        except Exception as e: 
            self._is_restoring = False
            print(f"Error restore: {e}")

    def on_engine_change(self, value):
        """Handler for Engine Mode segmented button."""
        from core.processor import EngineDispatcher
        
        # Internal mapping
        mapping = {
            self.locale.get("engine_auto"): "auto",
            self.locale.get("engine_cpu"): "cpu",
            self.locale.get("engine_gpu"): "gpu"
        }
        mode = mapping.get(value, "auto")
        EngineDispatcher.set_mode(mode)
        
        # Reset current task to stop previous backend operations
        self.cancel_current_task()
        
        # Re-trigger processing with new backend
        self.on_param_change()

    def cancel_current_task(self):
        """Interrupts currently running background task."""
        TaskManager.interrupt_all()
        self.status_label.configure(text=f"❌ Stopped", text_color="#e74c3c")
        self.after(2000, lambda: self.status_label.configure(text="", text_color="#2ecc71"))

    def on_param_change(self, *args):
        if self._is_restoring: return
        
        mode = self._get_logical(self.setting_mode_switch.get(), "setting_mode")
        
        if self.current_inventory_id is not None:
            e = self.image_manager.get_image(self.current_inventory_id)
            if e:
                if mode == "Individual":
                    # Save UI state directly to the specific image
                    e["params"] = self.capture_ui_state()
                    # Refresh preview for current image
                    self.process_inventory_image(e["pil_image"])
                else:
                    # Global mode: Update temporary global state
                    self.global_params = self.capture_ui_state()
                    
                    # If this image is NOT in the active global group, 
                    # we must NOT use the global UI values for its processing.
                    # Instead, we use its own stored params if they exist.
                    target_params = self.global_params if e.get("is_active_global", True) else (e["params"] if e["params"] else self.cached_defaults)
                    self._start_threaded_process("pil", e["pil_image"], params_override=target_params)
                
                # Update button visibility for current image
                self.update_reset_button_visibility(self.current_inventory_id)
        elif self.original_image_path: 
            # Non-inventory single image mode
            if mode == "Global":
                self.global_params = self.capture_ui_state()
            self.process_image()

    def select_inventory_image(self, image_id):
        mode = self._get_logical(self.setting_mode_switch.get(), "setting_mode")
        e = self.image_manager.get_image(image_id)
        
        # Task 52.1: Block selection if excluded in Global mode
        if mode == "Global" and e and not e.get("is_active_global", True):
            self.status_label.configure(text="⚠️ 활성 그룹에 포함되지 않은 이미지입니다.", text_color="#e74c3c")
            self.after(2000, lambda: self.status_label.configure(text=""))
            return

        self.current_inventory_id = image_id
        self.update_inventory_appearance()
        if e:
            if self._get_logical(self.setting_mode_switch.get(), "setting_mode") == "Individual":
                if e["params"]: 
                    self.restore_ui_state(e["params"])
                else: 
                    # CRITICAL FIX: Use copy of defaults, NOT current UI state
                    # which might belong to the previously selected image.
                    import copy
                    e["params"] = copy.deepcopy(self.cached_defaults)
                    self.restore_ui_state(e["params"])
            else: 
                self.process_inventory_image(e["pil_image"])

    def process_inventory_image(self, pil_image): 
        self._start_threaded_process("pil", pil_image)

    def process_image(self):
        if self.original_image_path: 
            self._start_threaded_process("path", self.original_image_path)

    def _start_threaded_process(self, source_type, source_data, params_override=None):
        # 1. Capture/Determine Parameters
        if params_override:
            params = params_override
        elif self.current_inventory_id is not None:
            e = self.image_manager.get_image(self.current_inventory_id)
            mode = self._get_logical(self.setting_mode_switch.get(), "setting_mode")
            if mode == "Individual" and e and e["params"]:
                params = e["params"]
            else:
                params = self.capture_ui_state() # Use current UI for global/single
        else:
            params = self.capture_ui_state()

        # 2. Change Detection Optimization
        state_key = (self.current_inventory_id if self.current_inventory_id is not None else source_data)
        if self.last_processed_params.get(state_key) == params:
            # OPTIMIZATION: If params matched, check if we have a valid cached result to display
            if self.current_inventory_id is not None:
                e = self.image_manager.get_image(self.current_inventory_id)
                if e and e.get("processed_image") is not None:
                    # Restore from inventory cache instead of re-processing
                    self.raw_pixel_image = e["processed_image"]
                    self.preview_image = e["preview_image"]
                    self.display_image()
                    self._update_secondary_ui(e["processed_image"])
                    return
            else:
                return # For non-inventory single image, just skip if params match

        if self._is_processing:
            self._pending_reprocess = True
            # New request invalidates ongoing one
            TaskManager.interrupt_all()
            return

        # Start New Task
        current_task_id = TaskManager.start_new_task()
        self.last_processed_params[state_key] = params.copy() # Update cache
        self.status_label.configure(text="⏳ Processing... (ESC to cancel)", text_color="#e67e22")

        def run():
            try:
                # 1. Get Base Image (RGBA)
                if source_type == "path":
                    img = Image.open(source_data).convert("RGBA")
                else:
                    img = source_data.convert("RGBA")
                
                self.original_size = img.size
                TaskManager.check(current_task_id)
                
                # 2. Check Background Removal Cache
                img_no_bg = None
                current_bg_params = {
                    "bg_mode": params.get("bg_mode", "None"),
                    "bg_seeds": params.get("bg_seeds", []),
                    "foreground_seeds": params.get("foreground_seeds", []) # Fixed name
                }
                
                entry = self.image_manager.get_image(self.current_inventory_id) if self.current_inventory_id is not None else None
                
                if entry and entry["bg_processed_image"] is not None and entry["last_bg_params"] == current_bg_params:
                    # Use Cache
                    img_no_bg = entry["bg_processed_image"]
                else:
                    # Execute Background Removal (Heavy Task)
                    img_no_bg = self.plugin_engine.execute_hook("PRE_PROCESS", img, params)
                    
                    bg_m = current_bg_params["bg_mode"]
                    if bg_m == "AI Auto":
                        img_no_bg = remove_background_ai(img_no_bg)
                    elif bg_m == "Interactive" and current_bg_params["bg_seeds"]:
                        img_no_bg = remove_background_interactive(img_no_bg, current_bg_params["bg_seeds"], current_bg_params["foreground_seeds"])
                    elif bg_m == "Classic":
                        img_no_bg = remove_background(img_no_bg, tolerance=40)
                    
                    # Update Cache
                    if entry:
                        entry["bg_processed_image"] = img_no_bg
                        entry["last_bg_params"] = current_bg_params.copy()

                TaskManager.check(current_task_id)

                # 3. Rest of the Pipeline (Light Tasks)
                raw = pixelate_image(img_no_bg, params["pixel_size"], 
                                     edge_enhance=params["edge_enhance"], 
                                     edge_sensitivity=params["edge_sensitivity"], 
                                     downsample_method=params["downsample_method"], 
                                     plugin_engine=self.plugin_engine, 
                                     plugin_params=params,
                                     task_id=current_task_id)

                if not raw: 
                    self.after(0, self._on_processing_complete, None)
                    return
                
                TaskManager.check(current_task_id)

                p_name = params["palette_mode"]
                if p_name == "USER CUSTOM":
                    p_name, p_param = "Custom_User", params["custom_colors"]
                elif p_name == "16-bit (4096 Colors)":
                    p_name, p_param = "Custom_16bit", None
                else:
                    p_param = params["color_count"]

                proc = apply_palette_unified(raw, p_name, custom_colors=p_param, dither=params["dither"],
                                            extract_policy=params["extract_policy"],
                                            mapping_policy=params["mapping_policy"],
                                            w_sat=params["rap_w_sat"], w_con=params["rap_w_con"], w_rar=params["rap_w_rar"],
                                            auto_optimal=params.get("auto_optimal", False),
                                            task_id=current_task_id)
                
                proc = self.plugin_engine.execute_hook("POST_PALETTE", proc, params)
                
                # Apply Native Grain Effect (at the small pixel scale)
                if params.get("grain_enabled"):
                    proc = apply_grain_effect(proc, params.get("grain_intensity", 15))
                
                if params["outline"]: proc = add_outline(proc)
                proc = self.plugin_engine.execute_hook("FINAL_IMAGE", proc, params)
                
                TaskManager.check(current_task_id)
                prev = upscale_for_preview(proc, self.original_size)
                
                self.after(0, self._on_processing_complete, proc, prev)
                
            except OperationCancelled:
                debug_log(f"[Engine] Task {current_task_id} cancelled.")
                self.after(0, self._on_processing_complete, None)
            except Exception as e: 
                debug_log(f"Thread error: {e}")
                self.after(0, self._on_processing_complete, None)

        threading.Thread(target=run, daemon=True).start()

    def _update_secondary_ui(self, proc):
        """Updates palette inspector and resolution label without re-rendering."""
        try:
            clrs = self.extract_used_colors(proc)
            self.palette_inspector.update_colors(clrs)
            rw, rh = proc.size if self._get_logical(self.mode_switch.get(), "save_mode") == "Pixelate" else self.original_size
            self.res_label.configure(text=f"{rw} x {rh}")
        except: pass

    def _on_processing_complete(self, proc, prev=None):
        self._is_processing = False
        
        # Restore interactive guide if in Manual mode, otherwise clear
        bg_mode = self._get_logical(self.option_bg_mode.get(), "bg_mode")
        if bg_mode == "Interactive":
            self.status_label.configure(text=self.locale.get("msg_interactive_tip"), text_color="#e67e22")
        else:
            self.status_label.configure(text="", text_color="#2ecc71")
        
        if proc:
            self.raw_pixel_image, self.preview_image = proc, prev
            
            # Save to inventory cache if applicable
            if self.current_inventory_id is not None:
                e = self.image_manager.get_image(self.current_inventory_id)
                if e:
                    e["processed_image"] = proc
                    e["preview_image"] = prev
            
            self.display_image()
            self.btn_save.configure(state="normal")
            self._update_secondary_ui(proc)
        else:
            if self.image_manager.count() == 0:
                self.clear_preview()
        
        if self._pending_reprocess: 
            self._pending_reprocess = False
            self._start_threaded_process(*self._pending_source)

    def on_palette_sort_change(self, mode):
        self.palette_sort_mode = mode
        if hasattr(self, "raw_pixel_image") and self.raw_pixel_image:
            clrs = self.extract_used_colors(self.raw_pixel_image)
            self.palette_inspector.update_colors(clrs)

    def extract_used_colors(self, pil_img):
        from core.palette import sort_colors
        tmp = pil_img.convert("RGB")
        colors = tmp.getcolors(maxcolors=4096)
        if not colors: return []
        freq_sorted = sorted(colors, key=lambda x: x[0], reverse=True)
        top_16 = [c[1] for c in freq_sorted[:16]]
        return sort_colors(top_16, self.palette_sort_mode)

    def display_image(self):
        if not self.preview_image: return
        params = self.capture_ui_state()
        render_img = self.plugin_engine.execute_hook("UI_PRE_RENDER", self.preview_image, params)
        cw, ch = self.preview_canvas.winfo_width(), self.preview_canvas.winfo_height()
        if cw <= 1: cw, ch = 800, 600
        iw, ih = render_img.size
        if self.preview_zoom == -1: self.preview_zoom = min(cw / iw, ch / ih) * 0.9
        nw, nh = int(iw * self.preview_zoom), int(ih * self.preview_zoom)
        disp = render_img.resize((nw, nh), Image.NEAREST)
        disp = self.plugin_engine.execute_hook("UI_POST_RENDER", disp, params)
        self.tk_preview = ImageTk.PhotoImage(disp)
        if self.canvas_image_id: self.preview_canvas.delete(self.canvas_image_id)
        self.canvas_image_id = self.preview_canvas.create_image(cw//2, ch//2, image=self.tk_preview, anchor="center")

    def on_preview_wheel(self, event):
        if not self.preview_image: return
        self.preview_zoom = max(0.1, min(20.0, self.preview_zoom * (1.1 if event.delta > 0 else 0.9)))
        self.display_image()

    def on_resize(self, event): 
        if self.preview_image: self.display_image()

    def toggle_magnifier(self):
        if self.mag_window is None or not self.mag_window.winfo_exists(): self.mag_window = MagnifierWindow(self)
        else: self.mag_window.focus()

    def update_magnifier(self, event):
        if not self.preview_image or not self.canvas_image_id or not (self.mag_window and self.mag_window.winfo_exists()): return
        cw, ch = self.preview_canvas.winfo_width(), self.preview_canvas.winfo_height()
        iw, ih = self.preview_image.size
        nw, nh = int(iw * self.preview_zoom), int(ih * self.preview_zoom)
        sx = int((event.x - (cw//2 - nw//2)) / self.preview_zoom)
        sy = int((event.y - (ch//2 - nh//2)) / self.preview_zoom)
        if 0 <= sx < iw and 0 <= sy < ih: 
            self.mag_window.update_zoom(self.preview_image, (sx, sy), self.mag_zoom)

    def save_image(self):
        p = self.original_image_path
        if self.current_inventory_id is not None:
            e = self.image_manager.get_image(self.current_inventory_id)
            if e: p = e["path"]
        if not p: return
        
        is_gif = p.lower().endswith('.gif')
        
        # Expanded file types for professional pixel art workflow
        types = [
            ("PNG Image", "*.png"),
            ("Targa Image (TGA)", "*.tga"),
            ("BMP Windows Bitmap", "*.bmp"),
            ("WebP Image", "*.webp"),
            ("JPEG Image", "*.jpg"),
            ("TIFF Image", "*.tiff")
        ]
        if is_gif:
            types.insert(0, ("GIF Animation", "*.gif"))
            
        f = filedialog.asksaveasfilename(parent=self, 
                                         defaultextension=(".gif" if is_gif else ".png"), 
                                         filetypes=types)
        if f:
            f = os.path.normpath(f)
            ext = f.lower()
            
            if is_gif and ext.endswith('.gif'):
                # GIF Animation Path
                self.btn_save.configure(text="GIF...", state="disabled")
                self.update()
                ch = self.option_palette.get()
                pn, pp = ("Custom_User", self.user_palette_colors_persistent) if ch == "USER CUSTOM" else (("Custom_16bit", None) if ch == "16-bit (4096 Colors)" else (ch, int(self.color_slider.get())))
                
                params = self.capture_ui_state()
                process_gif(p, f, int(self.slider_pixel.get()), pn, pp, 
                            dither=params["dither"], 
                            outline_enabled=params["outline"],
                            extract_policy=params["extract_policy"],
                            mapping_policy=params["mapping_policy"],
                            w_sat=params["rap_w_sat"],
                            w_con=params["rap_w_con"],
                            w_rar=params["rap_w_rar"],
                            auto_optimal=params["auto_optimal"])
                self.btn_save.configure(text=self.locale.get("sidebar_export"), state="normal")
            else:
                # Static Image Path
                final = self.raw_pixel_image if self._get_logical(self.mode_switch.get(), "save_mode") == "Pixelate" else self.preview_image
                
                # Handle alpha channel for formats that don't support it well (JPG, BMP, TGA)
                if ext.endswith(('.jpg', '.jpeg', '.bmp', '.tga')):
                    if final.mode == "RGBA":
                        # Create background (black by default for pixel art)
                        bg = Image.new("RGB", final.size, (0, 0, 0))
                        bg.paste(final, mask=final.split()[3])
                        final = bg
                    else:
                        final = final.convert("RGB")
                
                if ext.endswith('.png'):
                    from PIL.PngImagePlugin import PngInfo
                    metadata = PngInfo()
                    metadata.add_text("Pixlato:Params", json.dumps(self.capture_ui_state()))
                    final.save(f, pnginfo=metadata)
                else:
                    final.save(f)
                
                self.status_label.configure(text=f"✅ Saved: {os.path.basename(f)}")
                self.after(3000, lambda: self.status_label.configure(text=""))

    def add_image_to_inventory(self):
        paths = filedialog.askopenfilenames(parent=self, filetypes=[("Image", "*.png *.jpg *.jpeg *.webp *.gif *.bmp")])
        new_ids = []
        for pt in paths: new_ids.extend(self.image_manager.add_image(pt))
        for i in new_ids:
            e = self.image_manager.get_image(i)
            if e: self._create_inventory_item(e)
        self.update_inventory_count_label()
        if self.image_manager.count() > 0 and self.current_inventory_id is None: 
            self.select_inventory_image(self.image_manager.get_all()[0]["id"])

    def toggle_global_inclusion(self, iid, var):
        """Callback for the inventory inclusion checkbox."""
        e = self.image_manager.get_image(iid)
        if e:
            is_active = var.get()
            e["is_active_global"] = is_active
            self.update_inventory_appearance()
            
            # Task 52.3: If current image is excluded in Global mode, switch preview
            mode = self._get_logical(self.setting_mode_switch.get(), "setting_mode")
            if mode == "Global" and self.current_inventory_id == iid and not is_active:
                # Find another active image
                all_entries = self.image_manager.get_all()
                next_active = next((item for item in all_entries if item.get("is_active_global", True)), None)
                if next_active:
                    self.select_inventory_image(next_active["id"])
                else:
                    self.clear_preview()
            elif self.current_inventory_id == iid:
                self.on_param_change()

    def update_inventory_appearance(self):
        """Updates borders and checkbox visibility based on Setting Mode (Global/Individual)."""
        mode = self._get_logical(self.setting_mode_switch.get(), "setting_mode")
        
        for iid, frame in self.inventory_widgets.items():
            e = self.image_manager.get_image(iid)
            if not e: continue
            
            cb = self.inventory_active_checkboxes.get(iid)
            # Find labels to update their cursors too
            children = frame.winfo_children()
            labels = [c for c in children if isinstance(c, ctk.CTkLabel)]
            
            if mode == "Global":
                # Show checkbox in Global mode
                if cb: 
                    cb.place(x=5, y=5)
                    cb.lift()
                
                # Update border and background based on inclusion
                is_active = e.get("is_active_global", True)
                cursor = "hand2" if is_active else "arrow"
                fg = "#2a2a2a" if is_active else "#1a1a1a"
                
                frame.configure(fg_color=fg, cursor=cursor)
                for lbl in labels: lbl.configure(cursor=cursor)
                
                if is_active:
                    frame.configure(border_width=2, border_color="#2ecc71")
                else:
                    frame.configure(border_width=0)
            else:
                # Hide checkbox in Individual mode
                if cb: cb.place_forget()
                # Default appearance for Individual mode
                frame.configure(fg_color="#2a2a2a", cursor="hand2")
                for lbl in labels: lbl.configure(cursor="hand2")
                
                if self.current_inventory_id == iid:
                    frame.configure(border_width=2, border_color="#3498db")
                else:
                    frame.configure(border_width=0)

    def update_inventory_count_label(self):
        self.locale.register(self.label_inv_header, "inv_title", suffix=f" ({self.image_manager.count()}/256)")

    def _create_inventory_item(self, e):
        fid = e["id"]
        # Border width 2 for clear visual feedback
        item = ctk.CTkFrame(self.inventory_scroll, height=70, fg_color="#2a2a2a", corner_radius=8, border_width=0)
        item.pack(fill="x", pady=3, padx=2)
        item.pack_propagate(False)
        self.inventory_widgets[fid] = item
        
        # Inclusion Checkbox (Task 51.2) - Small and top-left
        var = ctk.BooleanVar(value=e.get("is_active_global", True))
        cb = ctk.CTkCheckBox(item, text="", width=18, height=18, checkbox_width=18, checkbox_height=18,
                             variable=var, command=lambda i=fid, v=var: self.toggle_global_inclusion(i, v))
        self.inventory_active_checkboxes[fid] = cb
        # Initially hidden, will be managed by update_inventory_appearance
        
        thumb = e["thumbnail"]
        ctk_t = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=(50, 50))
        tl = ctk.CTkLabel(item, image=ctk_t, text="")
        tl.pack(side="left", padx=5, pady=5)
        tl.image = ctk_t
        
        nl = ctk.CTkLabel(item, text=e["name"], font=("Arial", 10), anchor="w", justify="left", wraplength=100)
        nl.pack(side="left", fill="x", expand=True, padx=5)
        
        # Reset button: Restore default settings for this specific item
        rb = ctk.CTkButton(item, text="↻", width=24, height=24, fg_color="#34495e", 
                            hover_color="#2c3e50", command=lambda i=fid: self.reset_individual_params(i))
        # Don't place it yet, visibility will be handled by update_reset_button_visibility
        self.inventory_reset_buttons[fid] = rb
        ToolTip(rb, "Reset settings to defaults")
        
        # Initial visibility check (especially important for images with metadata)
        self.update_reset_button_visibility(fid, force_check=True)

        # Delete button: Robust click handling
        db = ctk.CTkButton(item, text="X", width=24, height=24, fg_color="#e74c3c", 
                            hover_color="#c0392b", command=lambda i=fid: self.remove_from_inventory(i))
        db.place(relx=1.0, rely=0, anchor="ne", x=-5, y=5)
        
        # Helper to select this item
        def select_this(event=None):
            self.select_inventory_image(fid)
            
        item.bind("<Button-1>", select_this)
        tl.bind("<Button-1>", select_this)
        nl.bind("<Button-1>", select_this)
        
        # Ensure checkbox is always on top (Task 51.5)
        cb.lift()

    def reset_individual_params(self, iid):
        """Restores default parameters for a specific inventory item and forces Individual mode."""
        print(f"DEBUG: Resetting ID: {iid}")
        
        mode = self._get_logical(self.setting_mode_switch.get(), "setting_mode")
        import copy
        defaults = self.get_hardcoded_defaults()

        if mode == "Global":
            # 1. Apply current global settings to ALL images to preserve state
            all_entries = self.image_manager.get_all()
            for entry in all_entries:
                entry["params"] = copy.deepcopy(self.global_params)
            
            # 2. Reset specific image
            self.image_manager.update_image_params(iid, copy.deepcopy(defaults))
            
            # 3. Use unified orchestrator to switch mode and sync UI
            self.current_inventory_id = iid
            self.set_ui_mode("Individual")
        else:
            # Already Individual, just reset
            self.image_manager.update_image_params(iid, copy.deepcopy(defaults))
            self.current_inventory_id = iid
            self.update_inventory_appearance()

        # 4. Apply visuals
        self.restore_ui_state(defaults)
        
        # 5. Hide the reset button since it's now back to defaults
        self.update_reset_button_visibility(iid, force_check=True)
        
        self.status_label.configure(text=f"↺ Reset ID {iid} (Switched to Individual)")
        self.after(2000, lambda: self.status_label.configure(text=""))

    def remove_from_inventory(self, iid):
        if iid in self.inventory_widgets:
            self.inventory_widgets[iid].destroy()
            del self.inventory_widgets[iid]
        
        # Clean up reset button reference
        if iid in self.inventory_reset_buttons:
            del self.inventory_reset_buttons[iid]
            
        # Clean up active checkbox reference
        if iid in self.inventory_active_checkboxes:
            del self.inventory_active_checkboxes[iid]
        
        self.image_manager.remove_image(iid)
        self.update_inventory_count_label()
        
        if self.current_inventory_id == iid:
            self.current_inventory_id = None
            if self.image_manager.count() > 0:
                # Select the first available item
                first_item = self.image_manager.get_all()[0]
                self.select_inventory_image(first_item["id"])
            else:
                self.clear_preview()

    def clear_preview(self):
        self.raw_pixel_image = None
        self.preview_image = None
        self.original_image_path = None
        self.current_inventory_id = None
        self.preview_canvas.delete("all")
        self.canvas_image_id = None
        self.res_label.configure(text="0 x 0")
        self.palette_inspector.update_colors([])
        self.btn_save.configure(state="disabled")

    def export_used_palette_gpl(self):
        if not self.raw_pixel_image: return
        colors = self.extract_used_colors(self.raw_pixel_image)
        if not colors: return
        f = filedialog.asksaveasfilename(parent=self, defaultextension=".gpl", filetypes=[("GIMP Palette", "*.gpl")], initialfile="pixlato_palette.gpl")
        if f:
            from core.palette import export_as_gpl
            if export_as_gpl(f, colors):
                self.status_label.configure(text="✅ GPL Exported")
                self.after(3000, lambda: self.status_label.configure(text=""))

    def batch_export(self):
        if self.image_manager.count() > 0: BatchExportWindow(self, self._start_batch_export_process)

    def _start_batch_export_process(self, od, fs, win, ss=False, sep=False, norm_settings=None):
        import concurrent.futures
        from collections import defaultdict
        import math
        g, mode, entries = self.capture_ui_state(), self._get_logical(self.setting_mode_switch.get(), "setting_mode"), self.image_manager.get_all()
        groups = defaultdict(list)
        for e in entries: groups[e["path"]].append(e)

        def process_frame_layered(e, target_params):
            p = target_params
            
            # 1. Background Removal
            current_bg_params = {
                "bg_mode": p.get("bg_mode", "None"),
                "bg_seeds": p.get("bg_seeds", []),
                "fg_seeds": p.get("fg_seeds", [])
            }
            
            if e.get("bg_processed_image") is not None and e.get("last_bg_params") == current_bg_params:
                img = e["bg_processed_image"]
            else:
                img = e["pil_image"].convert("RGBA")
                bg_m = current_bg_params["bg_mode"]
                if bg_m == "AI Auto":
                    from core.processor import remove_background_ai
                    img = remove_background_ai(img)
                elif bg_m == "Interactive" and current_bg_params["bg_seeds"]:
                    from core.processor import remove_background_interactive
                    img = remove_background_interactive(img, current_bg_params["bg_seeds"], current_bg_params["fg_seeds"])
                elif bg_m == "Classic":
                    img = remove_background(img, tolerance=40)
                
                e["bg_processed_image"] = img
                e["last_bg_params"] = current_bg_params.copy()
                
            # 2. Edge & Pixelation
            if p.get("edge_enhance"): 
                from core.processor import enhance_internal_edges
                img = enhance_internal_edges(img, p["edge_sensitivity"])
                
            sw, sh = max(1, img.size[0] // p["pixel_size"]), max(1, img.size[1] // p["pixel_size"])
            small = img.resize((sw, sh), resample=Image.BOX)
            
            # 3. Palette Application
            ch = p["palette_mode"]
            pn, pp = ("Custom_User", p["custom_colors"]) if ch == "USER CUSTOM" else (("Custom_16bit", None) if ch == "16-bit (4096 Colors)" else (ch, p.get("color_count", 16)))
            
            proc = apply_palette_unified(small, pn, pp, p.get("dither", True), 
                                         extract_policy=p.get("extract_policy", "Standard"), 
                                         mapping_policy=p.get("mapping_policy", "Classic"),
                                         w_sat=p.get("rap_w_sat", 0.4), w_con=p.get("rap_w_con", 0.3), w_rar=p.get("rap_w_rar", 0.3),
                                         auto_optimal=p.get("auto_optimal", False))
            
            if p.get("grain_enabled"): proc = apply_grain_effect(proc, p.get("grain_intensity", 15))
            
            # 4. Upscale back to Original Result Size
            orig_size = e["pil_image"].size
            base = proc.resize(orig_size, Image.NEAREST)
            
            # 5. [New Phase 50] Resolution Normalization
            if norm_settings and norm_settings.get("enabled"):
                target_res = (norm_settings["target_w"], norm_settings["target_h"])
                # Decide strategy based on relative size
                if target_res[0] > base.size[0] or target_res[1] > base.size[1]:
                    strat = norm_settings["upscale_strategy"]
                else:
                    strat = norm_settings["downscale_strategy"]
                
                base = normalize_image_geometry(base, target_res, strategy=strat)

            # 6. Outline & Layering
            outline_img = None
            if p.get("outline"):
                from PIL import ImageChops
                # Re-calculate outline on the potentially normalized base
                out_small = add_outline(proc)
                # Ensure outline is upscaled to match normalized base
                base_for_diff = proc.resize(orig_size, Image.NEAREST)
                diff = ImageChops.difference(out_small, proc)
                outline_img = diff.resize(orig_size, Image.NEAREST)
                
                final = out_small.resize(orig_size, Image.NEAREST)
                
                # Apply same normalization to final and outline if enabled
                if norm_settings and norm_settings.get("enabled"):
                    target_res = (norm_settings["target_w"], norm_settings["target_h"])
                    final = normalize_image_geometry(final, target_res, strategy=strat)
                    outline_img = normalize_image_geometry(outline_img, target_res, strategy=strat)
            else: 
                final = base
                
            return final, base, outline_img

        def task_individual(e, f):
            try:
                p = e["params"] if mode == "Individual" and e["params"] else g
                final, base, outline = process_frame_layered(e, p)
                bn = os.path.basename(e['name'])
                ext = f.lower()
                if sep and outline:
                    base.save(os.path.join(od, f"{bn}_base.png")); outline.save(os.path.join(od, f"{bn}_outline.png"))
                    return True, f"✅ Layers: {bn}"
                else:
                    # Handle alpha channel for specific formats
                    if ext in ["jpg", "jpeg", "bmp", "tga"]:
                        if final.mode == "RGBA":
                            bg = Image.new("RGB", final.size, (0, 0, 0))
                            bg.paste(final, mask=final.split()[3])
                            final = bg
                        else:
                            final = final.convert("RGB")
                    
                    save_path = os.path.join(od, f"{bn}_pixel.{ext}")
                    final.save(save_path)
                    return True, f"✅ {bn} ({f})"
            except Exception as ex: return False, f"❌ {e['name']}: {ex}"

        def task_gif_group(path, frames, custom_duration=None):
            try:
                processed_frames = []
                for e in frames:
                    p = e["params"] if mode == "Individual" and e["params"] else g
                    final, _, _ = process_frame_layered(e, p)
                    processed_frames.append(final.convert("P", palette=Image.ADAPTIVE))
                
                # Single merged GIF name
                save_path = os.path.join(od, "pixlato_merged_sequence.gif")
                dur = custom_duration if custom_duration else 100
                if processed_frames: 
                    processed_frames[0].save(save_path, save_all=True, append_images=processed_frames[1:], 
                                            duration=dur, loop=0, optimize=False)
                return True, f"✅ GIF Merged: {len(processed_frames)} frames"
            except Exception as ex: return False, f"❌ GIF: {ex}"

        def task_spritesheet(path, frames):
            try:
                processed = []
                for e in frames:
                    p = e["params"] if mode == "Individual" and e["params"] else g
                    final, _, _ = process_frame_layered(e, p); processed.append(final)
                if not processed: return False, "No frames"
                n = len(processed); cols = math.ceil(math.sqrt(n)); rows = math.ceil(n / cols); fw, fh = processed[0].size
                sheet = Image.new("RGBA", (cols * fw, rows * fh), (0,0,0,0))
                for i, img in enumerate(processed): r, c = divmod(i, cols); sheet.paste(img, (c * fw, r * fh))
                base_name = "pixlato_spritesheet"; sheet.save(os.path.join(od, f"{base_name}.png"))
                return True, f"✅ Sheet: {base_name}"
            except Exception as ex: return False, f"❌ Sheet: {ex}"

        def run():
            done = 0; tasks = []
            gif_dur = norm_settings.get("duration", 100) if norm_settings else 100
            if ss:
                tasks.append(("spritesheet", "merged", entries))
            for f in fs:
                if f == "GIF":
                    tasks.append(("gif", "merged", entries, gif_dur))
                else:
                    for e in entries: tasks.append(("individual", e, f))
            if sep and not tasks:
                for e in entries: tasks.append(("individual", e, "PNG"))
            total = len(tasks)
            if total == 0: 
                if win.winfo_exists(): self.after(0, lambda: win.btn_start.configure(state="normal", text="작업 완료"))
                return
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                fut = []
                for t in tasks:
                    if t[0] == "gif": fut.append(ex.submit(task_gif_group, t[1], t[2], t[3]))
                    elif t[0] == "spritesheet": fut.append(ex.submit(task_spritesheet, t[1], t[2]))
                    else: fut.append(ex.submit(task_individual, t[1], t[2]))
                
                for f in concurrent.futures.as_completed(fut):
                    done += 1
                    s, m = f.result()
                    if win.winfo_exists():
                        self.after(0, lambda m=m: win.log(m))
                        self.after(0, lambda d=done, tl=total: win.update_progress(d, tl))
            
            if win.winfo_exists():
                self.after(0, lambda: win.btn_start.configure(state="normal", text="작업 완료"))
        threading.Thread(target=run, daemon=True).start()

    def save_preset_dialog(self):
        dialog = ctk.CTkInputDialog(text="새로운 프리셋 이름을 입력하세요:", title="프리셋 저장")
        name = dialog.get_input()
        if name:
            self.presets[name] = self.capture_ui_state()
            try:
                with open(self.presets_path, "w", encoding="utf-8") as f: json.dump(self.presets, f, indent=4)
                self.option_presets.configure(values=[self.locale.get("preset_default")] + list(self.presets.keys()))
                self.option_presets.set(name)
            except Exception as e: print(f"Error saving preset: {e}")

class MagnifierWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent); self.title("🔍 Pixel Magnifier"); self.geometry("300x380"); self.attributes("-topmost", True); self.parent = parent
        self.display_label = ctk.CTkLabel(self, text="", width=280, height=280, fg_color="black", corner_radius=10); self.display_label.pack(pady=10, padx=10)
        f = ctk.CTkFrame(self, fg_color="transparent"); f.pack(pady=5, fill="x", padx=10)
        ctk.CTkButton(f, text="-", width=40, command=self.zoom_out).pack(side="left", padx=5)
        self.label_zoom = ctk.CTkLabel(f, text=f"{self.parent.mag_zoom}x", font=("Arial", 16, "bold")); self.label_zoom.pack(side="left", expand=True)
        ctk.CTkButton(f, text="+", width=40, command=self.zoom_in).pack(side="left", padx=5)
    def zoom_in(self):
        if self.parent.mag_zoom < 16: self.parent.mag_zoom += 1; self.label_zoom.configure(text=f"{self.parent.mag_zoom}x")
    def zoom_out(self):
        if self.parent.mag_zoom > 2: self.parent.mag_zoom -= 1; self.label_zoom.configure(text=f"{self.parent.mag_zoom}x")
    def update_zoom(self, img, pos, z):
        sz = max(1, int(280 / z)); sx, sy = pos; l, t = max(0, sx - sz // 2), max(0, sy - sz // 2); r, b = min(img.size[0], l + sz), min(img.size[1], t + sz)
        crop = img.crop((l, t, r, b)).resize((280, 280), Image.NEAREST); ci = ctk.CTkImage(light_image=crop, dark_image=crop, size=(280, 280))
        self.display_label.configure(image=ci, text=""); self.display_label.image = ci

if __name__ == "__main__": app = PixelApp(); app.mainloop()
