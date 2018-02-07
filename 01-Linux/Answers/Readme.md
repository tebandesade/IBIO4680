1. What is the grep command?
	The grep command is similar to a regular represion where your input is a textual pattern you want to search for. For example in the lab excercise, when you grep for gray, you saw how words containing gray where highlighted, even if the whole word wasn't gray.
2. What is the meaning of #! /bin/bash
	#! /bin/bash in the first line means what interpreter you want your code to be compile with. It is not only restricted to bash. For instance, if you wanted to use a specific version of python or perl, you would specify the interpreter in the first line.
3. How many users exist in the course server?
	If I log to 157.253.63.7, and run users I see six of them (one being vision). To get a more direct way of counting the users you can add a pipe to get the output of users as the input of wc -w, which counts. In other words, users | wc -w
	Also, you can check who is part of the server using cat /etc/passwd | grep -i /home/ 
4. 

5. Create a script for finding duplicate images based on their content (tip: hash or checksum) You may look in the internet for ideas, Do not forget to include the source of any code you use.
	See point5.py 
7. What is the disk size of the uncompressed dataset, How many images are in the directory 'BSR/BSDS500/data/images'?
	-rwxrwxrwx 1 root root 70763455 Jan 22  2013 BSR_bsds500.tgz
	It is 70763455 bytes 

	Running in data -> find . -name "*.jpg" -exec identify {} \;| wc -l
	the result is 500
8. What is their resolution, what is their format?
	Check format_resolution.txt for the answer
9. How many of them are in landscape orientation (opposed to portrait)?
	Portrait:  152	
	Landscape:  348
	To see code see check_landscape_portrait.py
10. Crop all images to make them square (256x256).
	Please see reshape.py to see the code and output see reshape.txt

