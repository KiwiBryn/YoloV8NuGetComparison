//---------------------------------------------------------------------------------
// Copyright (c) November 2024, devMobile Software - Azure Event Grid + YoloV8 file PoC
//
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU
// Affero General Public License as published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Affero General Public License for more details.
// You should have received a copy of the GNU Affero General Public License along with this program. 
// If not, see <https://www.gnu.org/licenses/>
//
//---------------------------------------------------------------------------------
using Microsoft.Extensions.Configuration;
#if GPURELEASE
   using Microsoft.ML.OnnxRuntime;
#endif
using Compunet.YoloV8;
using Compunet.YoloV8.Plotting;

using SixLabors.ImageSharp; 
using SixLabors.ImageSharp.PixelFormats;


namespace devMobile.IoT.YoloV8.Coprocessor.Detect.Image
{
   [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE0063:Use simple 'using' statement", Justification = "I prefer the old style")]
   class Program
   {
      static async Task Main()
      {
         Model.ApplicationSettings _applicationSettings;

         Console.WriteLine($"{DateTime.UtcNow:yy-MM-dd HH:mm:ss} Detect.dem-Compunet.Detect.Image starting");
#if RELEASE
         Console.WriteLine("RELEASE");
#else
         Console.WriteLine("DEBUG");
#endif

         try
         {
            // load the app settings into configuration
            var configuration = new ConfigurationBuilder()
                 .AddJsonFile("appsettings.json", false, true)
            .Build();

            _applicationSettings = configuration.GetSection("ApplicationSettings").Get<Model.ApplicationSettings>();

            Console.WriteLine($"{DateTime.UtcNow:yy-MM-dd HH:mm:ss.fff} YoloV8 Model load: {_applicationSettings.ModelPath}");

            YoloV8Builder builder = new YoloV8Builder();

            builder.UseOnnxModel(_applicationSettings.ModelPath);

#if GPURELEASE
            if (_applicationSettings.UseCuda)
            {
               Console.WriteLine($"{DateTime.UtcNow:yy-MM-dd HH:mm:ss.fff} Using CUDA");

               using (OrtCUDAProviderOptions cudaProviderOptions = new())
               {
                  Dictionary<string, string> optionKeyValuePairs = new()
                  {
                     //{ "gpu_mem_limit", ""},
                     //{ "arena_extend_strategy", "0: },
                     //{ "cudnn_conv_algo_search", "0"},
                     //{ "do_copy_in_default_stream", "1"},
                     //{ "cudnn_conv_use_max_workspace", "0"},
                     //{ "cudnn_conv1d_pad_to_nc1d" , "0"},
                     //{ "enable_cuda_graph", "0"},
                     { "device_id" , _applicationSettings.DeviceId.ToString()},
                  };

                  cudaProviderOptions.UpdateOptions(optionKeyValuePairs);

                  string options = cudaProviderOptions.GetOptions();

                  options = options.Replace(";", Environment.NewLine);

                  Console.WriteLine($"CUDA Options:");
                  Console.WriteLine(options);

                  builder.UseCuda(cudaProviderOptions);
               }
            }

            if (_applicationSettings.UseTensorrt)
            {
               Console.WriteLine($"{DateTime.UtcNow:yy-MM-dd HH:mm:ss.fff} Using TensorRT");

               using (OrtTensorRTProviderOptions tensorRToptions = new())
               {
                  Dictionary<string, string> optionKeyValuePairs = new()
                  {
                     //{ "trt_max_workspace_size", "2147483648" },                    
                     //{ "trt_max_partition_iterations", "1000" },
                     //{ "trt_min_subgraph_size", "1" },

                     //{ "trt_fp16_enable", "1" },
                     //{ "trt_int8_enable", "0" },

                     //{ "trt_int8_calibration_table_name", "" },
                     //{ "trt_int8_use_native_calibration_table", "0" },

                     //{ "trt_dla_enable", "1" },
                     //{ "trt_dla_core", "0" },

                     //{ "trt_timing_cache_enable", "1" },
                     //{ "trt_timing_cache_path", "timingcache/" },

                     { "trt_engine_cache_enable", "1" },
                     { "trt_engine_cache_path", "enginecache/" },

                     //{ "trt_dump_ep_context_model", "1" },
                     //{ "trt_ep_context_file_path", "embedengine/" },

                     //{ "trt_dump_subgraphs", "0" },
                     //{ "trt_force_sequential_engine_build", "0" },

                     { "device_id" , _applicationSettings.DeviceId.ToString()},
                  };

                  tensorRToptions.UpdateOptions(optionKeyValuePairs);

                  string options = tensorRToptions.GetOptions();

                  options = options.Replace(";", Environment.NewLine);

                  Console.WriteLine($"Tensor RT Options:");
                  Console.WriteLine(options);
 
                  builder.UseTensorrt(tensorRToptions);
               }
            }
#endif
            /*            
            builder.WithConfiguration(c =>
            {
               c.Confidence = 0.0f;
               c.IoU = 0.0f;
               c.KeepOriginalAspectRatio = false;
               c.SuppressParallelInference = false ;
            });
            */

            /*
            builder.WithSessionOptions(new Microsoft.ML.OnnxRuntime.SessionOptions()
            {
               EnableCpuMemArena
               EnableMemoryPattern
               EnableProfiling = true,
               ExecutionMode = ExecutionMode.
               GraphOptimizationLevel = GraphOptimizationLevel.
               InterOpNumThreads = 1,
               ProfileOutputPathPrefix = ""
               OptimizedModelFilePath = ""                
            });
            */

            using (var predictor = builder.Build())
            {
               using (var image = await SixLabors.ImageSharp.Image.LoadAsync<Rgba32>(_applicationSettings.ImageInputPath))
               {
                  Console.WriteLine($"{DateTime.UtcNow:yy-MM-dd HH:mm:ss.fff} Input image Width:{image.Width} Height:{image.Height} File:{_applicationSettings.ImageInputPath}");

                  var predictions = await predictor.DetectAsync(image);

                  Console.WriteLine($" {DateTime.UtcNow:yy-MM-dd HH:mm:ss.fff} YoloV8 Warmup Iterations:{_applicationSettings.IterationsWarmUp}");

                  for (var i = 1; i <= _applicationSettings.IterationsWarmUp; i++)
                  {
                     predictions = await predictor.DetectAsync(image);

                     //Console.WriteLine($"{DateTime.UtcNow:yy-MM-dd HH:mm:ss.fff} Warmup {i}");
                  }

                  Console.WriteLine($" {DateTime.UtcNow:yy-MM-dd HH:mm:ss.fff} YoloV8 Model detect start Iterations:{_applicationSettings.Iterations}");
                  DateTime start = DateTime.UtcNow;

                  for (var i = 0; i < _applicationSettings.Iterations; i++)
                  {
                     predictions = await predictor.DetectAsync(image);
                  }

                  DateTime finish = DateTime.UtcNow;
                  Console.WriteLine($" {finish:yy-MM-dd HH:mm:ss.fff} YoloV8 Model detect done");
                  TimeSpan duration = finish - start;
                  Console.WriteLine($" Average:{duration.TotalMilliseconds/_applicationSettings.Iterations:F0}mSec");

                  Console.WriteLine($" Boxes: {predictions.Boxes.Count()}");

                  foreach (var box in predictions.Boxes)
                  {
                     Console.WriteLine($"  Class {box.Class.Name} {(box.Confidence * 100.0):f1}% X:{box.Bounds.Left} Y:{box.Bounds.Right} Width:{box.Bounds.Width} Height:{box.Bounds.Height}");
                  }
                  Console.WriteLine();

                  Console.WriteLine($"{DateTime.UtcNow:yy-MM-dd HH:mm:ss.fff} Plot and save : {_applicationSettings.ImageOutputPath}");

                  using (var imageOutput = await predictions.PlotImageAsync(image))
                  {
                     Console.WriteLine($"{DateTime.UtcNow:yy-MM-dd HH:mm:ss.fff} Output image Width:{imageOutput.Width} Height:{imageOutput.Height}");

                     await imageOutput.SaveAsJpegAsync(_applicationSettings.ImageOutputPath);
                  }
               }
            }
         }
         catch (Exception ex)
         {
            Console.WriteLine($"{DateTime.UtcNow:yy-MM-dd HH:mm:ss} Application failure {ex}");
         }

         Console.WriteLine("Press enter to exit");
         Console.ReadLine();
      }
   }
}
