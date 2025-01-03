﻿//---------------------------------------------------------------------------------
// Copyright (c) August 2024, devMobile Software - Azure Event Grid + YoloV8 file PoC
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

using YoloDotNet;
using YoloDotNet.Extensions;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;


namespace devMobile.IoT.YoloV8.Detect.NickSwardh.Image
{
   internal class Program
   {
      [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE0063:Use simple 'using' statement", Justification = "I prefer the old style")]
      static async Task Main()
      {
         Model.ApplicationSettings _applicationSettings;

         Console.WriteLine($"{DateTime.UtcNow:yy-MM-dd HH:mm:ss} YoloV8.NickSwardh.Image starting");

         try
         {
            // load the app settings into configuration
            var configuration = new ConfigurationBuilder()
               .AddJsonFile("appsettings.json", false, true)
               .Build();

            _applicationSettings = configuration.GetSection("ApplicationSettings").Get<Model.ApplicationSettings>();

            Console.WriteLine($"{DateTime.UtcNow:yy-MM-dd HH:mm:ss.fff} YoloV8 Model load: {_applicationSettings.ModelPath}");

            using (var predictor = new Yolo(_applicationSettings.ModelPath, _applicationSettings.UseCuda))
            {
               using (var image = await SixLabors.ImageSharp.Image.LoadAsync<Rgba32>(_applicationSettings.ImageInputPath))
               {
                  Console.WriteLine($"{DateTime.UtcNow:yy-MM-dd HH:mm:ss.fff} Input image Width:{image.Width} Height:{image.Height} File:{_applicationSettings.ImageInputPath}");

                  var predictions = predictor.RunObjectDetection(image);

                  for (var i = 1; i <= _applicationSettings.IterationsWarmUp; i++)
                  {
                     predictions = predictor.RunObjectDetection(image);

                     Console.WriteLine($"{DateTime.UtcNow:yy-MM-dd HH:mm:ss.fff} Warmup {i}");
                  }

                  Console.WriteLine($" {DateTime.UtcNow:yy-MM-dd HH:mm:ss.fff} YoloV8 Model detect start");
                  DateTime start = DateTime.UtcNow;

                  for (int i = 0; i < _applicationSettings.Iterations; i += 1)
                  {
                     predictions = predictor.RunObjectDetection(image);
                  }

                  DateTime finish = DateTime.UtcNow;
                  Console.WriteLine($" {finish:yy-MM-dd HH:mm:ss.fff} YoloV8 Model detect done");
                  TimeSpan duration = finish - start;
                  Console.WriteLine($" Average:{duration.TotalMilliseconds/_applicationSettings.Iterations:f0}mSec");

                  Console.WriteLine($" Boxes: {predictions.Count}");

                  foreach (var predicition in predictions)
                  {
                     Console.WriteLine($"  Class {predicition.Label.Name} {(predicition.Confidence * 100.0):f1}% X:{predicition.BoundingBox.Left} Y:{predicition.BoundingBox.Right} Width:{predicition.BoundingBox.Width} Height:{predicition.BoundingBox.Height}");
                  }
                  Console.WriteLine();

                  Console.WriteLine($" {DateTime.UtcNow:yy-MM-dd HH:mm:ss.fff} Plot and save : {_applicationSettings.ImageOutputPath}");

                  image.Draw(predictions);

                  await image.SaveAsJpegAsync(_applicationSettings.ImageOutputPath);
               }
            }
         }
         catch (Exception ex)
         {
            Console.WriteLine($"{DateTime.UtcNow:yy-MM-dd HH:mm:ss} Application failure {ex.Message}", ex);
         }

         Console.WriteLine("Press enter to exit");
         Console.ReadLine();
      }
   }
}
