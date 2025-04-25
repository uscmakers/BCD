# tello_keyboard_control.py

from djitellopy import Tello
import pygame
import time

# --- New constant for move distance ---
MOVE_DISTANCE = 30 # Distance in cm for each move command (min 20)
YAW_ANGLE = 45     # Angle in degrees for each yaw command

def keyboard_control():
    """
    Initializes Pygame and Tello, then enters the main control loop using move commands.
    """
    pygame.init()
    pygame.display.set_caption("Tello Keyboard Control - Move API")
    # Create a small window, necessary for keyboard event capture
    screen = pygame.display.set_mode((200, 200))

    tello = Tello()

    try:
        tello.connect(False)

        # --- Remove speed dictionary, no longer needed ---
        # speed = {'LR': 0, 'FB': 0, 'UD': 0, 'Yaw': 0}

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # --- Single press actions ---
                    if event.key == pygame.K_t: # Takeoff
                        if not tello.is_flying:
                            print("Taking off...")
                            try:
                                tello.takeoff()
                            except Exception as e:
                                print(f"Takeoff failed: {e}")
                        else:
                            print("Already flying.")
                    elif event.key == pygame.K_l: # Land
                        if tello.is_flying:
                            print("Landing...")
                            try:
                                tello.land()
                            except Exception as e:
                                print(f"Landing failed: {e}")
                        else:
                            print("Already landed.")
                    elif event.key == pygame.K_e: # Emergency stop
                        print("EMERGENCY STOP")
                        tello.emergency()
                        running = False # Often good to stop after emergency
                    elif event.key == pygame.K_q: # Quit
                        print("Quitting...")
                        running = False

                    # --- Discrete movement commands (only if flying) ---
                    elif tello.is_flying:
                        if event.key == pygame.K_UP:
                            print(f"Move up {MOVE_DISTANCE} cm")
                            tello.move_up(MOVE_DISTANCE)
                        elif event.key == pygame.K_DOWN:
                            print(f"Move down {MOVE_DISTANCE} cm")
                            tello.move_down(MOVE_DISTANCE)
                        elif event.key == pygame.K_LEFT: # Yaw Left
                            print(f"Rotate counter-clockwise {YAW_ANGLE} degrees")
                            tello.rotate_counter_clockwise(YAW_ANGLE)
                        elif event.key == pygame.K_RIGHT: # Yaw Right
                            print(f"Rotate clockwise {YAW_ANGLE} degrees")
                            tello.rotate_clockwise(YAW_ANGLE)
                        elif event.key == pygame.K_w: # Forward
                            print(f"Move forward {MOVE_DISTANCE} cm")
                            tello.move_forward(MOVE_DISTANCE)
                        elif event.key == pygame.K_s: # Backward
                            print(f"Move back {MOVE_DISTANCE} cm")
                            tello.move_back(MOVE_DISTANCE)
                        elif event.key == pygame.K_a: # Left
                            print(f"Move left {MOVE_DISTANCE} cm")
                            tello.move_left(MOVE_DISTANCE)
                        elif event.key == pygame.K_d: # Right
                            print(f"Move right {MOVE_DISTANCE} cm")
                            tello.move_right(MOVE_DISTANCE)

                # --- Remove KEYUP handling, no longer needed ---
                # elif event.type == pygame.KEYUP:
                #     ...

            # --- Remove send_rc_control block ---
            # if tello.is_flying:
            #     tello.send_rc_control(...)

            # --- Optional small delay to prevent accidental double commands ---
            # time.sleep(0.1) # Adjust as needed, or remove

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up...")
        # Ensure landing and resource release on exit/error
        if tello.is_flying:
            try:
                print("Attempting to land before exit...")
                tello.land()
            except Exception as land_err:
                print(f"Landing attempt failed: {land_err}")
                print("Trying emergency stop...")
                tello.emergency() # Last resort
        tello.end()
        pygame.quit()
        print("Script finished.")

if __name__ == '__main__':
    keyboard_control()