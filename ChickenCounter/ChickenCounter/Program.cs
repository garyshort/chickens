using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Drawing;
using System.Drawing.Imaging;
using System.Configuration;
using System.Drawing.Drawing2D;

namespace ChickenCounter
{
    class Program
    {
        private static List<List<Point>> Clusters = new List<List<Point>>();
        private static int NeighbourThreshold = 0;
        private static int MergeThreshold = 0;
        private static int NoiseThreshold = 0;

        /// <summary>
        /// Main function and app start point.
        /// </summary>
        /// <param name="args">Path to image file</param>
        static void Main(string[] args)
        {
            // Set control plane
            SetControlPlane();

            // Ensure an image path is specified
            if (args.Length < 1)
            {
                Console.WriteLine("No image specified.");
                return;
            }

            // Load the image
            string path2Image = args[0];
            Console.WriteLine("Loading image {0}", path2Image);
            Bitmap imageIn;
            try { imageIn = new Bitmap(path2Image); }
            catch (FileNotFoundException)
            {
                Console.WriteLine("Can't find specified file!");
                return;
            }

            // Get a grayscale version of the image
            Console.WriteLine("Creating grayscale image.");
            Bitmap grayscale = MakeBitmapGrayscale(imageIn);

            // Release the full colour image
            imageIn.Dispose();
            imageIn = null;

            // Save the grayscale version so we can see it
            grayscale.Save("gray.bmp");

            // Use Otsu's method to find the threshold
            int threshold = GetThreshold(grayscale);

            // Convert the grayscale to binary using threshold
            Console.WriteLine(
                "Creating binary image at intensity: {0}",
                threshold);

            Bitmap binImage = Grayscale2Bin(grayscale, threshold);

            // Release the grayscale image
            grayscale.Dispose();
            grayscale = null;

            // Save the binary file so we can see it
            binImage.Save("binaryImage.bmp");

            // Cluster the points
            ClusterAllPoints(binImage);

            // Cull clusters with less than threshold pixels
            Console.WriteLine(
                "\nCulling clusters at noise threshold: {0}",
                NoiseThreshold);

            CullNoiseClusters(NoiseThreshold);

            // Merge clusters
            Console.WriteLine(
                "Merging clusters at merge threshold {0}.",
                MergeThreshold);

            MergeClusters();

            // Draw centres and number orginal image for control purposes
            DrawClusterCentres();
            NumberOriginalImage(args[0]);

            // How many chickens did we count?
            Console.WriteLine(
                "\nI counted {0} chickens in the image supplied.",
                Clusters.Count);
        }

        /// <summary>
        /// Append the cluster number to the cluster, (chicken), 
        /// on the original image
        /// </summary>
        /// <param name="filePath">Path to orginal image</param>
        private static void NumberOriginalImage(string filePath)
        {
            Bitmap original = (Bitmap)Bitmap.FromFile(filePath);
            Bitmap markedUp = new Bitmap(original);

            Graphics g = Graphics.FromImage(markedUp);
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;

            int ctr = 1;
            GetClusterCentrePoints().ForEach(point =>
            {
                g.DrawString(
                    ctr.ToString(),
                    new Font("Arial", 12, FontStyle.Bold),
                    Brushes.Red,
                    point);

                ctr++;
            });

            markedUp.Save("MarkedUpOriginal.bmp");
        }

        /// <summary>
        /// Read in the control values from the config file
        /// </summary>
        private static void SetControlPlane()
        {
            AppSettingsReader reader = new AppSettingsReader();
            NeighbourThreshold =
                (int)reader.GetValue("NeighbourThreshold", typeof(int));
            MergeThreshold =
                (int)reader.GetValue("MergeThreshold", typeof(int));
            NoiseThreshold =
                (int)reader.GetValue("NoiseThreshold", typeof(int));

        }

        /// <summary>
        /// Merge two clusters if their centres are less 
        /// than threshold value apart
        /// </summary>
        private static void MergeClusters()
        {
            List<List<Point>> delList = new List<List<Point>>();
            List<Point>[] c = Clusters.ToArray();
            int ptr1 = 0;
            int ptr2 = 1;

            while (ptr1 < c.Length)
            {
                while (ptr2 < c.Length)
                {
                    List<Point> l1 = c[ptr1];
                    List<Point> l2 = c[ptr2];
                    Point ctr1 = GetClusterCentrePoint(l1);
                    Point ctr2 = GetClusterCentrePoint(l2);
                    if (EuclideanDistanceBetween(ctr1, ctr2) < MergeThreshold)
                    {
                        l1.AddRange(l2);
                        delList.Add(l2);
                    }
                    ptr2++;
                }
                ptr1++;
                ptr2 = ptr1 + 1;
            }
            delList.ForEach(list => Clusters.Remove(list));
        }

        /// <summary>
        /// Draw cluster centres on the the binary image for control purposes
        /// </summary>
        private static void DrawClusterCentres()
        {
            // Get the centre points for each cluster
            List<Point> centres = GetClusterCentrePoints();
            DrawCentresOnImage(centres);
        }

        /// <summary>
        /// Draw a  list of centres on the image for control purposes
        /// </summary>
        /// <param name="centres">The list of centres to be drawn</param>
        private static void DrawCentresOnImage(List<Point> centres)
        {
            Bitmap original = (Bitmap)Bitmap.FromFile("binaryImage.bmp");
            Bitmap markedUp = new Bitmap(original);

            Graphics g = Graphics.FromImage(markedUp);
            Pen blackPen = new Pen(Color.Red, 2.0f);


            centres.ForEach(point =>
            {
                g.DrawEllipse(blackPen, point.X, point.Y, 7, 7);
            });

            markedUp.Save("MarkedUpBinary.bmp");
        }

        /// <summary>
        /// Return a list of centre points for the clusters
        /// </summary>
        /// <returns>A list of centre points</returns>
        private static List<Point> GetClusterCentrePoints()
        {
            List<Point> centres = new List<Point>();
            Clusters.ForEach(cluster =>
            {
                centres.Add(GetClusterCentrePoint(cluster));
            });
            return centres;
        }

        /// <summary>
        /// Returns the centre point for a given cluster
        /// </summary>
        /// <param name="cluster">The cluster</param>
        /// <returns>The centre point</returns>
        private static Point GetClusterCentrePoint(List<Point> cluster)
        {
            return new Point(
                (int)cluster.AsParallel().Average(p => p.X),
                (int)cluster.AsParallel().Average(p => p.Y));
        }

        /// <summary>
        /// Remove a cluster if it's size is less than threshold
        /// </summary>
        /// <param name="noiseThreshold">The threshold</param>
        private static void CullNoiseClusters(int noiseThreshold)
        {
            Clusters.RemoveAll(cluster =>
                cluster.Count < noiseThreshold);
        }

        /// <summary>
        /// Walk each pixel in the binary image and cluter it
        /// </summary>
        /// <param name="binImage">The image to walk</param>
        private static void ClusterAllPoints(Bitmap binImage)
        {
            int clustered = 0;
            int total = binImage.Width * binImage.Height;

            // Cluster points
            for (int i = 0; i < binImage.Width; i++)
            {
                for (int j = 0; j < binImage.Height; j++)
                {
                    if (binImage.GetPixel(i, j).ToArgb() == Color.White.ToArgb())
                    {
                        ClusterPoint(new Point(i, j));
                    }
                    clustered++;
                    UpdateProgress(clustered, total);
                }
            }
        }

        /// <summary>
        /// Let the user know how we're progressing
        /// </summary>
        /// <param name="clustered">Number of pixels clustered</param>
        /// <param name="total">Total pixels to cluster</param>
        private static void UpdateProgress(int clustered, int total)
        {
            string update = string.Format(
                "{0:n0} pixels of {1:n0} clustered.",
                clustered,
                total);

            Console.CursorLeft -= Console.CursorLeft;
            Console.Write(update);
        }

        /// <summary>
        /// Use a KNN algorithm - with a hueristic function 
        /// to dynamically amend K as we progress - to cluster
        /// a give point
        /// </summary>
        /// <param name="p">the point to cluster</param>
        private static void ClusterPoint(Point p)
        {
            List<Point> chosenCluster = null;
            double votesCast = 0d;

            // If this is the first point, it's the root of the first cluster
            if (Clusters.Count == 0)
            {
                List<Point> l = new List<Point>();
                l.Add(p);
                Clusters.Add(l);
            }
            else
            {
                // Otherwise iterate over all the clusters...
                Clusters.ForEach(cluster =>
                {
                    // Find all the points within PointThreshold distance
                    List<Point> votingPoints = cluster.FindAll(point =>
                        IsCloseTo(point, p));

                    // Sum the votes of the voting points
                    double totalVotes =
                        votingPoints.AsParallel().Sum(
                            aPoint => CalculateVoteOfPoint(aPoint, p));

                    // If this is the current max then this is the selected cluster
                    if (totalVotes > votesCast)
                    {
                        votesCast = totalVotes;
                        chosenCluster = cluster;
                    }
                });

                // After voting if there's a chosen cluster, add the point
                if (chosenCluster != null)
                {
                    chosenCluster.Add(p);
                }
                else
                {
                    // There's no close clusters, so start a new one
                    List<Point> l = new List<Point>();
                    l.Add(p);
                    Clusters.Add(l);
                }
            }
        }

        /// <summary>
        /// A linear function to weight the vote of each 
        /// neighbour based on Euclidean distance
        /// </summary>
        /// <param name="neighbour"></param>
        /// <param name="candidate"></param>
        /// <returns></returns>
        private static double CalculateVoteOfPoint(
            Point neighbour,
            Point candidate)
        {
            return 1 / EuclideanDistanceBetween(neighbour, candidate);
        }

        /// <summary>
        /// Answer if the neighbouring point is "close enough" to 
        /// the candidate point to be included in the vote
        /// </summary>
        /// <param name="candidate">The candidate point</param>
        /// <param name="neighbour">The neighbour point</param>
        /// <returns></returns>
        private static bool IsCloseTo(Point candidate, Point neighbour)
        {
            return EuclideanDistanceBetween(candidate, neighbour)
                <= NeighbourThreshold;
        }

        /// <summary>
        /// Answers the Euclidean distance between two points
        /// </summary>
        /// <param name="p1">The first point</param>
        /// <param name="p2">The second point</param>
        /// <returns></returns>
        private static double EuclideanDistanceBetween(Point p1, Point p2)
        {
            return Math.Sqrt(
                Math.Pow((p1.X - p2.X), 2) +
                Math.Pow((p1.Y - p2.Y), 2));
        }

        /// <summary>
        /// Answers a binary image from a given grayscale image
        /// </summary>
        /// <param name="grayscale">The grayscale image</param>
        /// <param name="threshold">The Otsu's Method threshold to use</param>
        /// <returns>A binary image</returns>
        private static Bitmap Grayscale2Bin(Bitmap grayscale, int threshold)
        {
            Bitmap bm = new Bitmap(grayscale);
            for (int i = 0; i < bm.Width; i++)
            {
                for (int j = 0; j < bm.Height; j++)
                {
                    Color c = bm.GetPixel(i, j);
                    int intensity = (c.R + c.G + c.B) / 3;
                    if (intensity <= threshold)
                    {
                        bm.SetPixel(i, j, Color.Black);
                    }
                    else
                    {
                        bm.SetPixel(i, j, Color.White);
                    }
                }
            }
            return bm;
        }

        /// <summary>
        /// Use Otsu's Method to answers the threshold that maximises 
        /// the inter-class difference between background and foreground 
        /// on a grayscale histogram
        /// </summary>
        /// <param name="image">The grayscale image</param>
        /// <returns>The threshold</returns>
        private static int GetThreshold(Bitmap image)
        {
            // Turn image into byte array
            byte[] imageBytes =
                (byte[])new ImageConverter().ConvertTo(image, typeof(byte[]));

            // Create a histogram of intensities
            int[] histogram = new int[256];
            int ptr = 0;
            while (ptr < imageBytes.Length)
            {
                int h = 0xFF & imageBytes[ptr];
                histogram[h]++;
                ptr++;
            }

            // calculate the intensity probabilities
            int totalPixels = imageBytes.Length;
            float sumOfIntensities = 0;
            for (int t = 0; t < 256; t++) sumOfIntensities += t * histogram[t];

            // Exhaustively interate each threshold to find the one that maiximizes the
            // intra class difference
            float sumOfBackgroundIntensities = 0;
            int backgroundWeight = 0;
            int foregroundWeight = 0;
            float maximumInterClassVariance = 0;
            int threshold = 0;
            for (int t = 0; t < 256; t++)
            {
                backgroundWeight += histogram[t];
                if (backgroundWeight == 0) continue;

                foregroundWeight = totalPixels - backgroundWeight;
                if (foregroundWeight == 0) break;

                sumOfBackgroundIntensities += (float)(t * histogram[t]);

                float backgroundMean =
                    sumOfBackgroundIntensities / backgroundWeight;

                float foregroundMean =
                    (sumOfIntensities - sumOfBackgroundIntensities) / foregroundWeight;

                // Calculate intra class variance
                float intraClassVariance =
                    (float)backgroundWeight * (float)foregroundWeight *
                        (backgroundMean - foregroundMean) *
                        (backgroundMean - foregroundMean);

                // Check if new maximum found
                if (intraClassVariance > maximumInterClassVariance)
                {
                    maximumInterClassVariance = intraClassVariance;
                    threshold = t;
                }
            }
            return threshold;
        }

        /// <summary>
        /// Optimised method to create a grayscale image from 
        /// a 256X256X3 colour image
        /// </summary>
        /// <param name="original">The original image</param>
        /// <returns>The grayscale image</returns>
        public static Bitmap MakeBitmapGrayscale(Bitmap original)
        {
            // Create a blank bitmap the same size as original
            Bitmap newBitmap = new Bitmap(original.Width, original.Height);

            // Get a graphics object from the new image
            Graphics g = Graphics.FromImage(newBitmap);

            // Create the grayscale ColorMatrix
            ColorMatrix colorMatrix = new ColorMatrix(
               new float[][]
               {
                   new float[] {.3f, .3f, .3f, 0, 0},
                   new float[] {.59f, .59f, .59f, 0, 0},
                   new float[] {.11f, .11f, .11f, 0, 0},
                   new float[] {0, 0, 0, 1, 0},
                   new float[] {0, 0, 0, 0, 1}
               }
            );

            // Create some image attributes
            ImageAttributes attributes = new ImageAttributes();

            // Set the color matrix attribute
            attributes.SetColorMatrix(colorMatrix);

            // Draw the original image on the new image
            // using the grayscale color matrix
            g.DrawImage(
                original,
                new Rectangle(0, 0, original.Width, original.Height),
                0,
                0,
                original.Width,
                original.Height,
                GraphicsUnit.Pixel,
                attributes);

            // Dispose the Graphics object
            g.Dispose();
            return newBitmap;
        }
    }
}
