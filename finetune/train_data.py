"""
Training data for LoRA fine-tuning.

RAW_TRAIN_DATA — list of (user_input, expected_json_output) tuples.
build_dataset() — formats into a HuggingFace Dataset ready for SFTTrainer.
"""

from datasets import Dataset
from llm_parser import UNIFIED_SYSTEM_PROMPT

RAW_TRAIN_DATA = [
    # ── direct_command ───────────────────────────────────────────────────────
    ("Cathey, turn on the light.",
     '{"type":"direct_command","device":"light","action":"turn_on","value":null,"reply":"Sure, turning on the light!"}'),
    ("Cathey, turn off the light.",
     '{"type":"direct_command","device":"light","action":"turn_off","value":null,"reply":"Got it, light is off."}'),
    ("Hello Cathey, switch on the light please.",
     '{"type":"direct_command","device":"light","action":"turn_on","value":null,"reply":"Turning on the light for you!"}'),
    ("Cathey, turn the light off.",
     '{"type":"direct_command","device":"light","action":"turn_off","value":null,"reply":"Light turned off."}'),
    ("Cathey, set the brightness to 50.",
     '{"type":"direct_command","device":"light","action":"set_brightness","value":50,"reply":"Setting brightness to 50%."}'),
    ("Cathey, dim the light to 30 percent.",
     '{"type":"direct_command","device":"light","action":"set_brightness","value":30,"reply":"Dimming the light to 30%."}'),
    ("Cathey, set brightness 80.",
     '{"type":"direct_command","device":"light","action":"set_brightness","value":80,"reply":"Brightness set to 80%."}'),
    ("Cathey, start RGB mode.",
     '{"type":"direct_command","device":"light","action":"rgb_cycle","value":null,"reply":"Starting RGB cycle mode!"}'),
    ("Cathey, turn on the AC.",
     '{"type":"direct_command","device":"ac","action":"turn_on","value":null,"reply":"Turning on the AC."}'),
    ("Cathey, turn off the AC.",
     '{"type":"direct_command","device":"ac","action":"turn_off","value":null,"reply":"AC turned off."}'),
    ("Cathey, set the AC to 24 degrees.",
     '{"type":"direct_command","device":"ac","action":"set_temperature","value":24,"reply":"Setting the AC to 24 degrees."}'),
    ("Cathey, set temperature to 22.",
     '{"type":"direct_command","device":"ac","action":"set_temperature","value":22,"reply":"AC temperature set to 22 degrees."}'),
    ("Cathey, set the air conditioner to 26.",
     '{"type":"direct_command","device":"ac","action":"set_temperature","value":26,"reply":"Setting the air conditioner to 26 degrees."}'),
    ("Cathey, AC temperature 20.",
     '{"type":"direct_command","device":"ac","action":"set_temperature","value":20,"reply":"AC set to 20 degrees."}'),
    ("Cathey, open the curtain.",
     '{"type":"direct_command","device":"curtain","action":"open","value":null,"reply":"Opening the curtain for you."}'),
    ("Cathey, close the curtain.",
     '{"type":"direct_command","device":"curtain","action":"close","value":null,"reply":"Closing the curtain."}'),
    ("Cathey, set curtain to 50 percent.",
     '{"type":"direct_command","device":"curtain","action":"set_position","value":50,"reply":"Setting curtain to 50%."}'),
    ("Cathey, open the window.",
     '{"type":"direct_command","device":"window","action":"open","value":null,"reply":"Opening the window."}'),
    ("Cathey, close the window.",
     '{"type":"direct_command","device":"window","action":"close","value":null,"reply":"Closing the window now."}'),
    ("Cathey, please close the window.",
     '{"type":"direct_command","device":"window","action":"close","value":null,"reply":"Sure, closing the window."}'),
    ("Cathey, open window halfway.",
     '{"type":"direct_command","device":"window","action":"set_position","value":50,"reply":"Setting window to 50% open."}'),
    ("Cathey, switch on the lights.",
'{"type":"direct_command","device":"light","action":"turn_on","value":null,"reply":"Lights are now on."}'),

("Cathey, switch off the lights.",
'{"type":"direct_command","device":"light","action":"turn_off","value":null,"reply":"Lights are now off."}'),
("Cathey, lights on.",
'{"type":"direct_command","device":"light","action":"turn_on","value":null,"reply":"Turning on the lights."}'),
("Cathey, lights off.",
'{"type":"direct_command","device":"light","action":"turn_off","value":null,"reply":"Turning off the lights."}'),
("Cathey, turn the lamp on.",
'{"type":"direct_command","device":"light","action":"turn_on","value":null,"reply":"Lamp is on."}'),
("Cathey, turn the lamp off.",
'{"type":"direct_command","device":"light","action":"turn_off","value":null,"reply":"Lamp is off."}'),
("Cathey, set brightness to 70 percent.",
'{"type":"direct_command","device":"light","action":"set_brightness","value":70,"reply":"Brightness set to 70%."}'),
("Cathey, set brightness to 20 percent.",
'{"type":"direct_command","device":"light","action":"set_brightness","value":20,"reply":"Brightness set to 20%."}'),
("Cathey, dim the light to 40 percent.",
'{"type":"direct_command","device":"light","action":"set_brightness","value":40,"reply":"Dimming light to 40%."}'),
("Cathey, increase brightness to 80 percent.",
'{"type":"direct_command","device":"light","action":"set_brightness","value":80,"reply":"Increasing brightness to 80%."}'),
("Cathey, decrease brightness to 25 percent.",
'{"type":"direct_command","device":"light","action":"set_brightness","value":25,"reply":"Reducing brightness to 25%."}'),
("Cathey, set light brightness 55.",
'{"type":"direct_command","device":"light","action":"set_brightness","value":55,"reply":"Brightness set to 55%."}'),
("Cathey, start RGB mode.",
'{"type":"direct_command","device":"light","action":"rgb_cycle","value":null,"reply":"Starting RGB mode."}'),
("Cathey, activate color cycle.",
'{"type":"direct_command","device":"light","action":"rgb_cycle","value":null,"reply":"Color cycle activated."}'),
("Cathey, turn on AC.",
'{"type":"direct_command","device":"ac","action":"turn_on","value":null,"reply":"AC is now on."}'),
("Cathey, turn off AC.",
'{"type":"direct_command","device":"ac","action":"turn_off","value":null,"reply":"AC is now off."}'),
("Cathey, AC on.",
'{"type":"direct_command","device":"ac","action":"turn_on","value":null,"reply":"Turning on the AC."}'),
("Cathey, AC off.",
'{"type":"direct_command","device":"ac","action":"turn_off","value":null,"reply":"Turning off the AC."}'),
("Cathey, set AC to 23 degrees.",
'{"type":"direct_command","device":"ac","action":"set_temperature","value":23,"reply":"Setting AC to 23 degrees."}'),
("Cathey, set AC to 26 degrees.",
'{"type":"direct_command","device":"ac","action":"set_temperature","value":26,"reply":"Setting AC to 26 degrees."}'),
("Cathey, set temperature to 24.",
'{"type":"direct_command","device":"ac","action":"set_temperature","value":24,"reply":"Temperature set to 24 degrees."}'),
("Cathey, make it 22 degrees.",
'{"type":"direct_command","device":"ac","action":"set_temperature","value":22,"reply":"Setting temperature to 22 degrees."}'),
("Cathey, cooling to 21.",
'{"type":"direct_command","device":"ac","action":"set_temperature","value":21,"reply":"Cooling set to 21 degrees."}'),
("Cathey, adjust AC to 25.",
'{"type":"direct_command","device":"ac","action":"set_temperature","value":25,"reply":"Adjusting AC to 25 degrees."}'),
("Cathey, air conditioner on.",
'{"type":"direct_command","device":"ac","action":"turn_on","value":null,"reply":"Air conditioner turned on."}'),
("Cathey, air conditioner off.",
'{"type":"direct_command","device":"ac","action":"turn_off","value":null,"reply":"Air conditioner turned off."}'),
("Cathey, open the curtain.",
'{"type":"direct_command","device":"curtain","action":"open","value":null,"reply":"Opening the curtain."}'),
("Cathey, close the curtain.",
'{"type":"direct_command","device":"curtain","action":"close","value":null,"reply":"Closing the curtain."}'),
("Cathey, open curtains.",
'{"type":"direct_command","device":"curtain","action":"open","value":null,"reply":"Opening curtains."}'),
("Cathey, close curtains.",
'{"type":"direct_command","device":"curtain","action":"close","value":null,"reply":"Closing curtains."}'),
("Cathey, curtain to 50 percent.",
'{"type":"direct_command","device":"curtain","action":"set_position","value":50,"reply":"Curtain set to 50%."}'),
("Cathey, set curtain to 30 percent.",
'{"type":"direct_command","device":"curtain","action":"set_position","value":30,"reply":"Curtain set to 30%."}'),
("Cathey, curtain halfway.",
'{"type":"direct_command","device":"curtain","action":"set_position","value":50,"reply":"Curtain set to halfway."}'),
("Cathey, curtain to 80 percent.",
'{"type":"direct_command","device":"curtain","action":"set_position","value":80,"reply":"Curtain set to 80%."}'),
("Cathey, fully open curtain.",
'{"type":"direct_command","device":"curtain","action":"open","value":null,"reply":"Curtain fully opened."}'),
("Cathey, fully close curtain.",
'{"type":"direct_command","device":"curtain","action":"close","value":null,"reply":"Curtain fully closed."}'),
("Cathey, open the window.",
'{"type":"direct_command","device":"window","action":"open","value":null,"reply":"Opening the window."}'),
("Cathey, close the window.",
'{"type":"direct_command","device":"window","action":"close","value":null,"reply":"Closing the window."}'),
("Cathey, window open.",
'{"type":"direct_command","device":"window","action":"open","value":null,"reply":"Window opened."}'),
("Cathey, window close.",
'{"type":"direct_command","device":"window","action":"close","value":null,"reply":"Window closed."}'),
("Cathey, set window to 50 percent.",
'{"type":"direct_command","device":"window","action":"set_position","value":50,"reply":"Window set to 50%."}'),
("Cathey, open window halfway.",
'{"type":"direct_command","device":"window","action":"set_position","value":50,"reply":"Window halfway open."}'),
("Cathey, window to 25 percent.",
'{"type":"direct_command","device":"window","action":"set_position","value":25,"reply":"Window set to 25%."}'),
("Cathey, window to 70 percent.",
'{"type":"direct_command","device":"window","action":"set_position","value":70,"reply":"Window set to 70%."}'),
("Cathey, fully open window.",
'{"type":"direct_command","device":"window","action":"open","value":null,"reply":"Window fully opened."}'),
("Cathey, fully close window.",
'{"type":"direct_command","device":"window","action":"close","value":null,"reply":"Window fully closed."}'),
    # ── needs_clarification ──────────────────────────────────────────────────
    ("Cathey, I feel cold.",
     '{"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"],"reply":"Would you like me to close the window or raise the AC temperature?"}'),
    ("Cathey, I feel a little bit cold.",
     '{"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"],"reply":"Would you like me to close the window or raise the AC temperature?"}'),
    ("Cathey, it is freezing in here.",
     '{"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"],"reply":"Would you like me to close the window or raise the AC temperature?"}'),
    ("Cathey, I am a bit chilly.",
     '{"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"],"reply":"Would you like me to close the window or raise the AC temperature?"}'),
    ("Cathey, I feel hot.",
     '{"type":"needs_clarification","question":"Would you like me to open the window or lower the AC temperature?","options":["open_window","lower_ac_temperature"],"reply":"Would you like me to open the window or lower the AC temperature?"}'),
    ("Cathey, it is too warm in here.",
     '{"type":"needs_clarification","question":"Would you like me to open the window or lower the AC temperature?","options":["open_window","lower_ac_temperature"],"reply":"Would you like me to open the window or lower the AC temperature?"}'),
    ("Cathey, it is stuffy.",
     '{"type":"needs_clarification","question":"Would you like me to open the window or lower the AC temperature?","options":["open_window","lower_ac_temperature"],"reply":"Would you like me to open the window or lower the AC temperature?"}'),
    ("Cathey, it is a bit dark.",
     '{"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"],"reply":"Would you like me to turn on the light or open the curtain?"}'),
    ("Cathey, it is dark in here.",
     '{"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"],"reply":"Would you like me to turn on the light or open the curtain?"}'),
    ("Cathey, this room is too dark.",
     '{"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"],"reply":"Would you like me to turn on the light or open the curtain?"}'),
    ("Cathey, I can barely see anything.",
     '{"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"],"reply":"Would you like me to turn on the light or open the curtain?"}'),
    ("Cathey, it is too bright.",
     '{"type":"needs_clarification","question":"Would you like me to dim the light or close the curtain?","options":["dim_light","close_curtain"],"reply":"Would you like me to dim the light or close the curtain?"}'),
    ("Cathey, the light is too strong.",
     '{"type":"needs_clarification","question":"Would you like me to dim the light or close the curtain?","options":["dim_light","close_curtain"],"reply":"Would you like me to dim the light or close the curtain?"}'),
    ("Cathey, it's kinda cold.",
'{"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"],"reply":"Would you like me to close the window or raise the AC temperature?"}'),

("Cathey, I'm feeling a bit chilly.",
'{"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"],"reply":"Would you like me to close the window or raise the AC temperature?"}'),

("Cathey, this place is freezing.",
'{"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"],"reply":"Would you like me to close the window or raise the AC temperature?"}'),

("Cathey, it's too hot.",
'{"type":"needs_clarification","question":"Would you like me to open the window or lower the AC temperature?","options":["open_window","lower_ac_temperature"],"reply":"Would you like me to open the window or lower the AC temperature?"}'),

("Cathey, it's really warm in here.",
'{"type":"needs_clarification","question":"Would you like me to open the window or lower the AC temperature?","options":["open_window","lower_ac_temperature"],"reply":"Would you like me to open the window or lower the AC temperature?"}'),

("Cathey, I feel sweaty.",
'{"type":"needs_clarification","question":"Would you like me to open the window or lower the AC temperature?","options":["open_window","lower_ac_temperature"],"reply":"Would you like me to open the window or lower the AC temperature?"}'),

("Cathey, it's stuffy here.",
'{"type":"needs_clarification","question":"Would you like me to open the window or lower the AC temperature?","options":["open_window","lower_ac_temperature"],"reply":"Would you like me to open the window or lower the AC temperature?"}'),

("Cathey, I need fresh air.",
'{"type":"needs_clarification","question":"Would you like me to open the window or lower the AC temperature?","options":["open_window","lower_ac_temperature"],"reply":"Would you like me to open the window or lower the AC temperature?"}'),

("Cathey, it's too dark.",
'{"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"],"reply":"Would you like me to turn on the light or open the curtain?"}'),

("Cathey, I can’t see well.",
'{"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"],"reply":"Would you like me to turn on the light or open the curtain?"}'),

("Cathey, it's gloomy.",
'{"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"],"reply":"Would you like me to turn on the light or open the curtain?"}'),

("Cathey, too much light here.",
'{"type":"needs_clarification","question":"Would you like me to dim the light or close the curtain?","options":["dim_light","close_curtain"],"reply":"Would you like me to dim the light or close the curtain?"}'),

("Cathey, it's blinding.",
'{"type":"needs_clarification","question":"Would you like me to dim the light or close the curtain?","options":["dim_light","close_curtain"],"reply":"Would you like me to dim the light or close the curtain?"}'),

("Cathey, lights are too strong.",
'{"type":"needs_clarification","question":"Would you like me to dim the light or close the curtain?","options":["dim_light","close_curtain"],"reply":"Would you like me to dim the light or close the curtain?"}'),

("Cathey, do something with the light.",
'{"type":"needs_clarification","question":"Would you like me to turn off the light or dim it?","options":["turn_off_light","dim_light"],"reply":"Would you like me to turn off the light or dim it?"}'),

("Cathey, fix the lighting.",
'{"type":"needs_clarification","question":"Would you like me to turn off the light or dim it?","options":["turn_off_light","dim_light"],"reply":"Would you like me to turn off the light or dim it?"}'),

("Cathey, make it less bright.",
'{"type":"needs_clarification","question":"Would you like me to dim the light or turn it off?","options":["dim_light","turn_off_light"],"reply":"Would you like me to dim the light or turn it off?"}'),

("Cathey, make it brighter.",
'{"type":"needs_clarification","question":"Would you like me to increase brightness or turn on more lights?","options":["increase_brightness","turn_on_light"],"reply":"Would you like me to increase brightness or turn on more lights?"}'),

("Cathey, open it.",
'{"type":"needs_clarification","question":"Do you want me to open the window or the curtain?","options":["open_window","open_curtain"],"reply":"Do you want me to open the window or the curtain?"}'),

("Cathey, close it.",
'{"type":"needs_clarification","question":"Do you want me to close the window or the curtain?","options":["close_window","close_curtain"],"reply":"Do you want me to close the window or the curtain?"}'),

("Cathey, turn it off.",
'{"type":"needs_clarification","question":"Do you want me to turn off the light or the AC?","options":["turn_off_light","turn_off_ac"],"reply":"Do you want me to turn off the light or the AC?"}'),

("Cathey, turn it on.",
'{"type":"needs_clarification","question":"Do you want me to turn on the light or the AC?","options":["turn_on_light","turn_on_ac"],"reply":"Do you want me to turn on the light or the AC?"}'),

("Cathey, make it better.",
'{"type":"needs_clarification","question":"Would you like me to adjust lighting or temperature?","options":["adjust_lighting","adjust_temperature"],"reply":"Would you like me to adjust lighting or temperature?"}'),

("Cathey, something feels off.",
'{"type":"needs_clarification","question":"Is it lighting, temperature, or airflow you want to adjust?","options":["adjust_lighting","adjust_temperature","adjust_airflow"],"reply":"Is it lighting, temperature, or airflow you want to adjust?"}'),

("Cathey, this room sucks.",
'{"type":"needs_clarification","question":"Would you like me to adjust lighting or temperature?","options":["adjust_lighting","adjust_temperature"],"reply":"Would you like me to adjust lighting or temperature?"}'),

("Cathey, this place is annoying.",
'{"type":"needs_clarification","question":"What would you like to change: lighting or temperature?","options":["adjust_lighting","adjust_temperature"],"reply":"What would you like to change: lighting or temperature?"}'),

("Cathey, I hate this lighting.",
'{"type":"needs_clarification","question":"Would you like me to dim the light or turn it off?","options":["dim_light","turn_off_light"],"reply":"Would you like me to dim the light or turn it off?"}'),

("Cathey, this AC is too much.",
'{"type":"needs_clarification","question":"Would you like me to lower or turn off the AC?","options":["lower_ac_temperature","turn_off_ac"],"reply":"Would you like me to lower or turn off the AC?"}'),

("Cathey, make it cooler.",
'{"type":"needs_clarification","question":"Do you want me to lower the AC temperature or open the window?","options":["lower_ac_temperature","open_window"],"reply":"Do you want me to lower the AC temperature or open the window?"}'),

("Cathey, make it warmer.",
'{"type":"needs_clarification","question":"Do you want me to raise the AC temperature or close the window?","options":["raise_ac_temperature","close_window"],"reply":"Do you want me to raise the AC temperature or close the window?"}'),
    # hard cases: colloquial / emotional
    ("Cathey, fuck this light.",
     '{"type":"needs_clarification","question":"Would you like me to turn off the light or dim it?","options":["turn_off_light","dim_light"],"reply":"Would you like me to turn off the light or dim it?"}'),
    ("Cathey, this light is annoying.",
     '{"type":"needs_clarification","question":"Would you like me to turn off the light or dim it?","options":["turn_off_light","dim_light"],"reply":"Would you like me to turn off the light or dim it?"}'),
    ("Cathey, I hate this light.",
     '{"type":"needs_clarification","question":"Would you like me to turn off the light or dim it?","options":["turn_off_light","dim_light"],"reply":"Would you like me to turn off the light or dim it?"}'),
    ("Cathey, make this room lively.",
     '{"type":"needs_clarification","question":"Would you like me to turn on the RGB cycle or open the curtain?","options":["rgb_cycle","open_curtain"],"reply":"Would you like me to turn on the RGB cycle or open the curtain?"}'),
    ("Cathey, it is boring in here.",
     '{"type":"needs_clarification","question":"Would you like me to turn on the RGB cycle or open the curtain?","options":["rgb_cycle","open_curtain"],"reply":"Would you like me to turn on the RGB cycle or open the curtain?"}'),
    ("Cathey, spice up this room.",
     '{"type":"needs_clarification","question":"Would you like me to turn on the RGB cycle or open the curtain?","options":["rgb_cycle","open_curtain"],"reply":"Would you like me to turn on the RGB cycle or open the curtain?"}'),
    ("Cathey, I am not comfortable.",
     '{"type":"needs_clarification","question":"Could you describe what feels uncomfortable — temperature, lighting, or something else?","options":["adjust_temperature","adjust_lighting"],"reply":"Could you describe what feels uncomfortable — temperature, lighting, or something else?"}'),

    # ── general_qa ───────────────────────────────────────────────────────────
    ("Cathey, how do I eat an apple?",
     '{"type":"general_qa","answer":"Wash it first, then eat it."}'),
    ("Cathey, how can I eat an apple?",
     '{"type":"general_qa","answer":"Rinse it under water and eat it directly, or peel it first if you prefer."}'),
    ("Cathey, how do I boil an egg?",
'{"type":"general_qa","answer":"Place the egg in boiling water for about 6-10 minutes depending on how firm you want it."}'),

("Cathey, how long should I cook pasta?",
'{"type":"general_qa","answer":"Most pasta takes about 8-12 minutes to cook. Check the package instructions for exact timing."}'),

("Cathey, how do I make tea?",
'{"type":"general_qa","answer":"Boil water, pour it over tea leaves or a tea bag, and let it steep for a few minutes."}'),

("Cathey, how do I store leftovers?",
'{"type":"general_qa","answer":"Store leftovers in an airtight container and refrigerate them within two hours of cooking."}'),

("Cathey, how long can cooked chicken stay in the fridge?",
'{"type":"general_qa","answer":"Cooked chicken can usually be stored safely in the fridge for 3 to 4 days."}'),

("Cathey, how do I clean a microwave?",
'{"type":"general_qa","answer":"Heat a bowl of water with lemon for a few minutes, then wipe the inside clean with a cloth."}'),

("Cathey, how do I clean the floor?",
'{"type":"general_qa","answer":"Sweep or vacuum first, then mop using a suitable floor cleaner and warm water."}'),

("Cathey, how often should I do laundry?",
'{"type":"general_qa","answer":"It depends on usage, but most people do laundry once or twice a week."}'),

("Cathey, how do I remove a bad smell from the fridge?",
'{"type":"general_qa","answer":"Clean the fridge and place baking soda inside to absorb odors."}'),

("Cathey, how do I unclog a sink?",
'{"type":"general_qa","answer":"Try using a plunger or pour a mixture of baking soda and vinegar followed by hot water."}'),

("Cathey, how do I make rice?",
'{"type":"general_qa","answer":"Rinse the rice, then cook it with water in a 1:2 ratio until the water is absorbed."}'),

("Cathey, how long does rice last in the fridge?",
'{"type":"general_qa","answer":"Cooked rice can be stored in the fridge for up to 3 to 4 days."}'),

("Cathey, how do I keep vegetables fresh?",
'{"type":"general_qa","answer":"Store vegetables in the fridge and keep them dry, using breathable bags if possible."}'),

("Cathey, how do I cut an onion without crying?",
'{"type":"general_qa","answer":"Chill the onion before cutting or use a sharp knife to reduce irritation."}'),

("Cathey, how do I make scrambled eggs?",
'{"type":"general_qa","answer":"Whisk eggs, cook them on low heat while stirring gently until soft and fluffy."}'),

("Cathey, how do I clean glass windows?",
'{"type":"general_qa","answer":"Use a glass cleaner or vinegar solution and wipe with a microfiber cloth."}'),

("Cathey, how do I save electricity at home?",
'{"type":"general_qa","answer":"Turn off unused devices, use energy-efficient appliances, and reduce heating or cooling usage."}'),

("Cathey, how do I keep my room cool?",
'{"type":"general_qa","answer":"Close curtains during the day and use fans or ventilation to keep air circulating."}'),

("Cathey, how do I keep my room warm?",
'{"type":"general_qa","answer":"Close windows, use insulation like curtains, and maintain steady heating."}'),

("Cathey, how do I get rid of dust?",
'{"type":"general_qa","answer":"Regularly wipe surfaces with a damp cloth and vacuum frequently."}'),

("Cathey, how do I clean a sofa?",
'{"type":"general_qa","answer":"Vacuum the surface and use a fabric cleaner suitable for your sofa material."}'),

("Cathey, how do I organize my room?",
'{"type":"general_qa","answer":"Declutter items, group similar things together, and use storage containers or shelves."}'),

("Cathey, how do I remove stains from carpet?",
'{"type":"general_qa","answer":"Blot the stain, then use a carpet cleaner or mild detergent solution."}'),

("Cathey, how do I dry clothes faster?",
'{"type":"general_qa","answer":"Use proper airflow, hang clothes with space between them, or use a dryer if available."}'),

("Cathey, how do I clean a bathroom?",
'{"type":"general_qa","answer":"Use disinfectant cleaner on surfaces, scrub thoroughly, and rinse with water."}'),

("Cathey, how do I prevent mold?",
'{"type":"general_qa","answer":"Keep areas dry, ensure good ventilation, and clean moisture-prone surfaces regularly."}'),

("Cathey, how do I store bread?",
'{"type":"general_qa","answer":"Store bread in a cool, dry place or freeze it for longer storage."}'),

("Cathey, how do I keep food warm?",
'{"type":"general_qa","answer":"Cover the food and use low heat or insulated containers to retain warmth."}'),

("Cathey, how do I wash fruits properly?",
'{"type":"general_qa","answer":"Rinse fruits under running water and gently rub the surface before eating."}'),

("Cathey, how do I reduce noise at home?",
'{"type":"general_qa","answer":"Use rugs, curtains, and soft materials to absorb sound and reduce noise."}'),
    # hard case: mentions dish/fridge → general knowledge, NOT device control
    ("Cathey, can I still eat this dish after a night in the fridge?",
     '{"type":"general_qa","answer":"Yes, most cooked food is safe for up to 3-4 days in the fridge."}'),
    ("Cathey, is it safe to reheat leftover rice?",
     '{"type":"general_qa","answer":"Yes, reheat it thoroughly until steaming. Avoid reheating more than once."}'),
    ("Cathey, how long does milk last in the fridge?",
     '{"type":"general_qa","answer":"Typically 5-7 days after opening if kept below 4 degrees Celsius."}'),
    ("Cathey, what time is it?",
     '{"type":"general_qa","answer":"I do not have access to real-time data. Please check your phone."}'),
    ("Cathey, what is the capital of France?",
     '{"type":"general_qa","answer":"The capital of France is Paris."}'),
    ("Cathey, tell me a joke.",
     '{"type":"general_qa","answer":"Why do programmers prefer dark mode? Because light attracts bugs!"}'),
    ("Cathey, how do I make coffee?",
     '{"type":"general_qa","answer":"Add ground coffee to a filter, pour hot water over it, and let it drip."}'),
    ("Cathey, is it going to rain today?",
     '{"type":"general_qa","answer":"I do not have weather access. Please check a weather app."}'),
    ("Cathey, how do I remove a stain from clothes?",
     '{"type":"general_qa","answer":"Rinse with cold water immediately, then apply detergent and wash normally."}'),
    ("Cathey, how many calories are in an egg?",
     '{"type":"general_qa","answer":"A large egg contains about 70-80 calories."}'),
    ("Cathey, can I drink tap water?",
     '{"type":"general_qa","answer":"It depends on your location. Check your local water quality report."}'),

    # ── invalid (no Nova prefix) ─────────────────────────────────────────────
    ("Hello.",                       '{"type":"invalid"}'),
    ("Turn on the light.",           '{"type":"invalid"}'),
    ("How are you today?",           '{"type":"invalid"}'),
    ("It is cold.",                  '{"type":"invalid"}'),
    ("What is going on?",            '{"type":"invalid"}'),
    ("Open the window.",             '{"type":"invalid"}'),
    ("Hey, can you help me?",        '{"type":"invalid"}'),
    ("I feel hot.",                  '{"type":"invalid"}'),
    ("Thanks.",                      '{"type":"invalid"}'),
    ("Bob, turn on the light.",      '{"type":"invalid"}'),
    ("Siri, turn on the light.",     '{"type":"invalid"}'),
    ("No, I feel cold.",             '{"type":"invalid"}'),
]


def build_dataset(tokenizer) -> Dataset:
    """
    Format RAW_TRAIN_DATA into a HuggingFace Dataset using the model's chat template.
    SFTTrainer computes loss only on the assistant turn (response masking is automatic).
    """
    def _fmt(inp: str, out: str) -> str:
        msgs = [
            {"role": "system",    "content": UNIFIED_SYSTEM_PROMPT},
            {"role": "user",      "content": f'Text: "{inp}"\nReturn JSON only.'},
            {"role": "assistant", "content": out},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    return Dataset.from_dict({"text": [_fmt(i, o) for i, o in RAW_TRAIN_DATA]})
