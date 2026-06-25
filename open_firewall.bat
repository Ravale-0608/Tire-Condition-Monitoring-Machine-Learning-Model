@echo off
echo Adding Windows Firewall rule for Tire Scanner (port 8000)...
netsh advfirewall firewall add rule name="TireScanner-8000" dir=in action=allow protocol=TCP localport=8000
echo Done. You can now access the app from your phone.
pause
