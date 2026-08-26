#!/bin/sh
# Reset the InfiRay thermal USB camera (0bda:5840) when it wedges after an
# unclean exit (Ctrl+Z, SIGKILL, crash). Safe to run anytime.
for d in /sys/bus/usb/devices/*; do
  [ -f "$d/idVendor" ] || continue
  if [ "$(cat "$d/idVendor")" = "0bda" ] && [ "$(cat "$d/idProduct" 2>/dev/null)" = "5840" ]; then
    echo "Resetting thermal cam at $d"
    echo 0 | sudo tee "$d/authorized" >/dev/null; sleep 1
    echo 1 | sudo tee "$d/authorized" >/dev/null; sleep 2
    echo "Done."; exit 0
  fi
done
echo "Thermal camera 0bda:5840 not found."; exit 1
