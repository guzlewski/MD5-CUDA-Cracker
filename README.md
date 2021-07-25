# MD5-CUDA-Cracker
Simple program for bruteforce MD5 hashes written in CUDA.

## Usage
```
.\MD5-CUDA-Cracker.exe [options] hash alphabet

Positional arguments:
hash            input md5 hash to crack [Required]
alphabet        input alphabet to generate combinations [Required]

Optional arguments:
-h --help       show this help message and exit
--min           starting lenght of combinations, 1 if not supplied
--max           maximum lenght of combinations, 8 if not supplied
--blocks        number of blocks in CUDA grid, 2048 if not supplied
--threads       number of threads in CUDA block, 512 if not supplied
```

## Examples
```
.\MD5-CUDA-Cracker.exe 5d41402abc4b2a76b9719d911017c592 qwertyuiopasdfghjklzxcvbnm --max 5

Cracking hash 5d41402abc4b2a76b9719d911017c592
with alphabet qwertyuiopasdfghjklzxcvbnm
alphabet length 26
combination minimum size 1
combination maximum size 5
blocks in CUDA grid 2048
threads in CUDA block 512
Found matching combination in 0.757277 s.
Recovered data: hello
Result is valid.
```
   
```
.\MD5-CUDA-Cracker.exe 25d55ad283aa400af464c76d713c07ad 1234567890

Cracking hash 25d55ad283aa400af464c76d713c07ad
with alphabet 1234567890
alphabet length 10
combination minimum size 1
combination maximum size 8
blocks in CUDA grid 2048
threads in CUDA block 512
Found matching combination in 0.948342 s.
Recovered data: 12345678
Result is valid.
```

## License
All Rights Reserved  
Copyright (c) 2021 guzlewski  
  
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN  
THE SOFTWARE.
