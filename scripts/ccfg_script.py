# --- START OF FILE ccfg_script.py ---
import logging
import sys
import traceback
import gradio as gr
from modules import scripts, script_callbacks
from functools import partial
from typing import Any

# We'll import the patching class from the new file
from CCFG.nodes_ccfg import CCFGNode

# --- Helper function to generate float choices for XYZ Grid ---
def get_float_choices_from_range(min_val: float, max_val: float, step: float):
    """Generates a list of float choices for an XYZ Grid axis."""
    choices = []
    current = min_val
    # Add a small tolerance to ensure max_val is included due to float precision
    while current <= max_val + (step / 2.0):
        choices.append(round(current, 8)) # Round to avoid float precision issues
        current += step
    return choices

class CCFGScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.enabled = False
        self.tau = 1.0 # Default value for tau
        self.ccfg_node_instance = CCFGNode() # Instantiate the node logic

    sorting_priority = 15.3 # Slightly different from CFG-Zero to avoid conflict

    def title(self):
        return "CCFG-Lite Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Toggle CCFG-Lite guidance. Applies adaptive scaling to the guidance vector.</i></p>")
            enabled = gr.Checkbox(label="Enable CCFG-Lite", value=self.enabled)
            tau_slider = gr.Slider(
                minimum=0.0,
                maximum=10.0,
                step=0.1,
                value=self.tau,
                label="Tau (Adaptive Sensitivity)",
                info="Higher values attenuate strong guidance more aggressively."
            )

        enabled.change(
            lambda x: self.update_enabled(x),
            inputs=[enabled],
            outputs=None
        )
        tau_slider.change(
            lambda x: self.update_tau(x),
            inputs=[tau_slider],
            outputs=None
        )
        # Store controls for process_before_every_sampling
        self.ui_controls = [enabled, tau_slider]
        return self.ui_controls

    def update_enabled(self, value):
        self.enabled = value
        logging.debug(f"CCFG-Lite: Enabled toggled to: {self.enabled}")

    def update_tau(self, value):
        self.tau = value
        logging.debug(f"CCFG-Lite: Tau updated to: {self.tau}")

    def process_before_every_sampling(self, p, *args, **kwargs):
        # args will contain the values from self.ui_controls in order
        if len(args) >= 2:
            self.enabled = args[0]
            self.tau = args[1]
        elif len(args) == 1:
            self.enabled = args[0]
            logging.warning("CCFG-Lite: Not enough arguments for tau, using current value.")
        else:
            logging.warning("CCFG-Lite: Not enough arguments provided, using current values.")

        # Handle XYZ Grid
        xyz_settings = getattr(p, "_ccfg_xyz", {})
        if "enabled" in xyz_settings:
            self.enabled = xyz_settings["enabled"].lower() == "true"
        if "tau" in xyz_settings:
            try:
                self.tau = float(xyz_settings["tau"])
            except ValueError:
                logging.warning(f"CCFG-Lite: Invalid tau value from XYZ Grid: {xyz_settings['tau']}")

        # Ensure we have the base unet for operations or restore it
        if hasattr(p, '_original_unet_before_ccfg_lite'):
            p.sd_model.forge_objects.unet = p._original_unet_before_ccfg_lite.clone()
        else:
            p._original_unet_before_ccfg_lite = p.sd_model.forge_objects.unet.clone()
        
        unet_to_patch = p.sd_model.forge_objects.unet # This is now a fresh clone

        if not self.enabled:
            # If it was previously enabled and patched by us, ensure model_sampler_post_cfg_function is cleared.
            if hasattr(unet_to_patch, "_ccfg_lite_patched"):
                unet_to_patch.set_model_sampler_post_cfg_function(None, "ccfg_guidance")
                delattr(unet_to_patch, "_ccfg_lite_patched")
            p.sd_model.forge_objects.unet = unet_to_patch
            if "ccfg_lite_enabled" in p.extra_generation_params:
                del p.extra_generation_params["ccfg_lite_enabled"]
            if "ccfg_lite_tau" in p.extra_generation_params:
                del p.extra_generation_params["ccfg_lite_tau"]
            logging.debug(f"CCFG-Lite: Disabled. UNet restored.")
            return

        logging.debug(f"CCFG-Lite: Enabling with Tau: {self.tau}")
        
        # Pass the tau setting to the patch method
        patched_unet = self.ccfg_node_instance.patch(
            unet_to_patch,
            tau=self.tau
        )[0] # patch returns a tuple (model,)
        
        p.sd_model.forge_objects.unet = patched_unet
        setattr(p.sd_model.forge_objects.unet, "_ccfg_lite_patched", True) # Mark it

        p.extra_generation_params.update({
            "ccfg_lite_enabled": True,
            "ccfg_lite_tau": self.tau,
        })
        logging.debug(f"CCFG-Lite: Enabled: {self.enabled}, Tau: {self.tau}. UNet Patched.")
        return

# --- XYZ Grid Integration ---
def ccfg_set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_ccfg_xyz"):
        p._ccfg_xyz = {}
    p._ccfg_xyz[field] = str(x) # XYZ grid sends strings

def make_ccfg_axis_on_xyz_grid():
    xyz_grid = None
    for script_data in scripts.scripts_data:
        if script_data.script_class.__module__ in ("xyz_grid.py", "xy_grid.py"):
            xyz_grid = script_data.module
            break

    if xyz_grid is None:
        logging.warning("CCFG-Lite: XYZ Grid script not found.")
        return

    if any(x.label.startswith("(CCFG-Lite)") for x in xyz_grid.axis_options):
        logging.info("CCFG-Lite: XYZ Grid options already registered.")
        return
        
    ccfg_options = [
        xyz_grid.AxisOption(
            label="(CCFG-Lite) Enabled",
            type=str,
            apply=partial(ccfg_set_value, field="enabled"),
            choices=lambda: ["True", "False"]
        ),
        xyz_grid.AxisOption(
            label="(CCFG-Lite) Tau",
            type=float,
            apply=partial(ccfg_set_value, field="tau"),
            # Use our custom helper function here
            choices=lambda: get_float_choices_from_range(0.0, 10.0, 0.5) 
        ),
    ]
    xyz_grid.axis_options.extend(ccfg_options)
    logging.info("CCFG-Lite: XYZ Grid options successfully registered.")

def on_ccfg_before_ui():
    try:
        make_ccfg_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(
            f"[-] CCFG-Lite Script: Error setting up XYZ Grid options:\n{error}",
            file=sys.stderr,
        )

script_callbacks.on_before_ui(on_ccfg_before_ui)
# --- END OF FILE ccfg_script.py ---
