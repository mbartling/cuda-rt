/* Written 2012 by Matthias S. Benkmann
 *
 * The author hereby waives all copyright and related rights to the contents
 * of this example file (example_arg.cc) to the extent possible under the law.
 */

/**
 * @file
 * @brief Demonstrates handling various types of option arguments (required, numeric,...) with
   no dependency on the C++ standard library (only C lib).
 *
 * @include example_arg.cc
 */

#define TINYOBJLOADER_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "optionparser.h"
#include "bvh.h"
#include "scene.h"
#include "raytracer.h"
#include "tiny_obj_loader.h"


struct Arg: public option::Arg
{
  static void printError(const char* msg1, const option::Option& opt, const char* msg2)
  {
    fprintf(stderr, "%s", msg1);
    fwrite(opt.name, opt.namelen, 1, stderr);
    fprintf(stderr, "%s", msg2);
  }

  static option::ArgStatus Unknown(const option::Option& option, bool msg)
  {
    if (msg) printError("Unknown option '", option, "'\n");
    return option::ARG_ILLEGAL;
  }

  static option::ArgStatus Required(const option::Option& option, bool msg)
  {
    if (option.arg != 0)
      return option::ARG_OK;

    if (msg) printError("Option '", option, "' requires an argument\n");
    return option::ARG_ILLEGAL;
  }

  static option::ArgStatus NonEmpty(const option::Option& option, bool msg)
  {
    if (option.arg != 0 && option.arg[0] != 0)
      return option::ARG_OK;

    if (msg) printError("Option '", option, "' requires a non-empty argument\n");
    return option::ARG_ILLEGAL;
  }

  static option::ArgStatus Numeric(const option::Option& option, bool msg)
  {
    char* endptr = 0;
    if (option.arg != 0 && strtol(option.arg, &endptr, 10)){};
    if (endptr != option.arg && *endptr == 0)
      return option::ARG_OK;

    if (msg) printError("Option '", option, "' requires a numeric argument\n");
    return option::ARG_ILLEGAL;
  }
};

void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

 enum  optionIndex { UNKNOWN, HELP, INPUT, NUMERIC, lORIENTATION, lPOSITION, lCOLOR, NONEMPTY, OUTPUT };
const option::Descriptor usage[] = {
  { UNKNOWN, 0,"", "",        Arg::Unknown, "USAGE: example_arg [options]\n\n"
                                            "Options:" },
  { HELP,    0,"", "help",    Arg::None,    "  \t--help  \tPrint usage and exit." },
  //{ OPTIONAL,0,"o","optional",Arg::Optional,"  -o[<arg>], \t--optional[=<arg>]"
  //                                          "  \tTakes an argument but is happy without one." },
  { INPUT,0,"i","input",Arg::Required,"  -i <arg>, \t--input=<arg>  \tInput file required." },
  { NUMERIC, 0,"n","numeric", Arg::Numeric, "  -n <num>, \t--numeric=<num>  \tRequires a number as argument." },
  { NONEMPTY,0,"1","nonempty",Arg::NonEmpty,"  -1 <arg>, \t--nonempty=<arg>"
                                            "  \tCan NOT take the empty string as argument." },
  { lORIENTATION, 0,"d","orientation", Arg::Required, "  -d <float,float,float>, \t--orientation=<float,float,float>  \tRequires 3 floats as argument." },
  { lPOSITION, 0,"p","position", Arg::Required, "  -p <float,float,float>, \t--position=<float,float,float>  \tRequires 3 floats as argument." },
  { lCOLOR, 0,"c","color", Arg::Required, "  -c <float,float,float>, \t--color=<float,float,float>  \tRequires 3 floats as argument." },
  { OUTPUT, 0,"o","output", Arg::Required, "  -o <arg>, \t--output=<arg>  \tOutput file argument required." },

  { UNKNOWN, 0,"", "",        Arg::None,
   "\nExamples:\n"
   "  example_arg --unknown -o -n10 \n"
   "  example_arg -o -n10 file1 file2 \n"
   "  example_arg -nfoo file1 file2 \n"
   "  example_arg --optional -- file1 file2 \n"
   "  example_arg --optional file1 file2 \n"
   "  example_arg --optional=file1 file2 \n"
   "  example_arg --optional=  file1 file2 \n"
   "  example_arg -o file1 file2 \n"
   "  example_arg -ofile1 file2 \n"
   "  example_arg -unk file1 file2 \n"
   "  example_arg -r -- file1 \n"
   "  example_arg -r file1 \n"
   "  example_arg --required \n"
   "  example_arg --required=file1 \n"
   "  example_arg --nonempty= file1 \n"
   "  example_arg --nonempty=foo --numeric=999 --optional=bla file1 \n"
   "  example_arg -1foo \n"
   "  example_arg -1 -- \n"
   "  example_arg -1 \"\" \n"
  },
  { 0, 0, 0, 0, 0, 0 } 
};

int main(int argc, char* argv[])
{
  string sceneName = "", outputFile = "", mtlFile = "";
  vector<string> x;
  float arr[3];
  Vec3f color = Vec3f(1.f,1.f,1.f), position = Vec3f(1.f,1.f,1.f), orientation = Vec3f(1.f,1.f,1.f);
  string::size_type sz;

  argc-=(argc>0); argv+=(argc>0); // skip program name argv[0] if present
  option::Stats stats(usage, argc, argv);

#ifdef __GNUC__
    // GCC supports C99 VLAs for C++ with proper constructor calls.
  option::Option options[stats.options_max], buffer[stats.buffer_max];
#else
    // use calloc() to allocate 0-initialized memory. It's not the same
    // as properly constructed elements, but good enough. Obviously in an
    // ordinary C++ program you'd use new[], but this file demonstrates that
    // TLMC++OP can be used without any dependency on the C++ standard library.
  option::Option* options = (option::Option*)calloc(stats.options_max, sizeof(option::Option));
  option::Option* buffer  = (option::Option*)calloc(stats.buffer_max,  sizeof(option::Option));
#endif

  option::Parser parse(usage, argc, argv, options, buffer);

  if (parse.error())
    return 1;

  if (options[HELP] || argc == 0)
  {
    int columns = getenv("COLUMNS")? atoi(getenv("COLUMNS")) : 80;
    option::printUsage(fwrite, stdout, usage, columns);
    return 0;
  }

  for (int i = 0; i < parse.optionsCount(); ++i)
  {
    option::Option& opt = buffer[i];
    fprintf(stdout, "Argument #%d is ", i);
    switch (opt.index())
    {
      case HELP:
        // not possible, because handled further above and exits the program
      //case OPTIONAL:
      //  if (opt.arg)
      //    fprintf(stdout, "--optional with optional argument '%s'\n", opt.arg);
      //  else
      //    fprintf(stdout, "--optional without the optional argument\n");
      //  break;
      case INPUT:
	{
	    fprintf(stdout, "--required with argument '%s'\n", opt.arg);
            sceneName = opt.arg;
            int found = sceneName.find_last_of("/");
            mtlFile = sceneName.substr(0,found+1);
	    fprintf(stdout, "mtlFile = %s\n", mtlFile.c_str());
	}
        break;
      case lORIENTATION:
        {
	    fprintf(stdout, "--light orientation with argument '%s'\n", opt.arg);
            split(opt.arg, ',', x);
            for(int i=0; i < x.size(); i++) {
                arr[i] = stof(x[i], &sz);
                fprintf(stdout, "options for orientation are %f \n", arr[i]);
            }
            orientation = Vec3f(arr[0], arr[1], arr[2]);
        }
        break;
      case lPOSITION:
        {
	    fprintf(stdout, "--light position with argument '%s'\n", opt.arg);
            split(opt.arg, ',', x);
            for(int i=0; i < x.size(); i++) {
                arr[i] = stof(x[i], &sz);
            }
            position = Vec3f(arr[0], arr[1], arr[2]);
        }
        break;
      case lCOLOR:
        {
	    fprintf(stdout, "--light color with argument '%s'\n", opt.arg);
            split(opt.arg, ',', x);
            for(int i=0; i < x.size(); i++) {
                arr[i] = stof(x[i], &sz);
                //fprintf(stdout, "options for color are %f \n", arr[i]);
            }
            color = Vec3f(arr[0], arr[1], arr[2]);
        }
        break;
      case NUMERIC:
        fprintf(stdout, "--numeric with argument '%s'\n", opt.arg);
        break;
      case NONEMPTY:
        fprintf(stdout, "--nonempty with argument '%s'\n", opt.arg);
        break;
      case OUTPUT:
        {
	    fprintf(stdout, "--output with argument '%s'\n", opt.arg);
            outputFile = opt.arg;
        }
        break;
      case UNKNOWN:
        // not possible because Arg::Unknown returns ARG_ILLEGAL
        // which aborts the parse with an error
        break;
    }
    x.clear(); //remove floats from previous parsing
  }

  for (int i = 0; i < parse.nonOptionsCount(); ++i)
    fprintf(stdout, "Non-option argument #%d is %s\n", i, parse.nonOption(i));

  /*build the tree */
  //Scene_h scene(1024,1024, 1);
  //scene.LoadObj(sceneName);
  //bvh(scene);

  RayTracer rayTracer;
  rayTracer.LoadObj(sceneName, mtlFile);
  Light_h hLight;
  hLight.color = color;
  hLight.position = position;
  hLight.orientation = orientation;
  rayTracer.setHostLight(hLight);
  
  rayTracer.setUpDevice();
  rayTracer.run();
  rayTracer.pullRaytracedImage();
  rayTracer.writeImage(outputFile);
 
}
