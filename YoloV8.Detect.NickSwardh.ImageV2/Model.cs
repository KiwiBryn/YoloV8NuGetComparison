//---------------------------------------------------------------------------------
// Copyright (c) March 2024, devMobile Software - Azure Event Grid + YoloV8 file PoC
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
namespace devMobile.IoT.YoloV8.Detect.NickSwardh.Image.Model
{
   public class ApplicationSettings
   {
         public required string ImageInputPath { get; set; }

         public required string ImageOutputPath { get; set; }

         public required string ModelPath { get; set; }

         public required string ImageProprocessedPath { get; set; }

         public int IterationsWarmUp { get; set; }

         public int Iterations { get; set; }

         public bool UseCuda { get; set; }

         public int GpuId { get; set; }
      }
}
