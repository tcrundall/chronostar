#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
  if (argc != 3 || strcmp(argv[1], "-h") == 0)
  {
    printf("usage: ./arguments arg1 arg2\n");
    return 1;
  }

  printf("%d arguments\n", argc);

  int i = 0;

  while(i < argc)
    printf("Argument %d: %s\n", i++, argv[i]); //argv[i] gets
                                               //evaluated first

  return 0;
}
