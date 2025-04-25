import asyncio
import time
from mavsdk import System
import serial.tools.list_ports

async def run():
    print("Starting drone connection attempt...")
    
    # List available ports for reference
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        print(p)
    
    # Create drone object
    drone = System(mavsdk_server_address="serial:///dev/cu.usbserial-0001", port=57600)
    
    try:
        # Use your CP2102 device with the correct format and baud rate
        # macOS has both /dev/tty.* and /dev/cu.* - the cu.* is often more reliable
        
        print(f"Attempting to connect using: {drone.mavsdk_server_address}")
        await drone.connect()
        
        print("Waiting for drone to connect...")
        connection_timeout = 15  # seconds
        start_time = time.time()
        connection_successful = False
        
        async for state in drone.core.connection_state():
            if state.is_connected:
                print(f"Drone discovered after {time.time() - start_time:.1f} seconds!")
                connection_successful = True
                break
            
            if time.time() - start_time > connection_timeout:
                print("Connection attempt timed out!")
                return
                
            print(".", end="", flush=True)
            await asyncio.sleep(1)
            
        if not connection_successful:
            print("Failed to connect to drone!")
            return

        print("\nFetching drone information...")
        try:
            async for info in drone.info.identification():
                print(f"Drone ID: {info}")
                break
        except Exception as e:
            print(f"Couldn't get drone info: {e}")
        
        print("Arming drone...")
        await drone.action.arm()
        print("Drone armed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        
    print("Mission complete")

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
