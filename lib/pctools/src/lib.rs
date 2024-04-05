use numpy::{
    ndarray::{Array1, Array2, ArrayView2, Axis, Zip},
    PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
};
use pyo3::prelude::*;

fn fps_sampling(
    points: ArrayView2<f32>,
    n_samples: usize,
    start_coord: Array1<f32>,
) -> Array2<f32> {
    let p = points.nrows();
    let c = points.ncols();

    // Validate start_coord dimensions
    if start_coord.len() != c {
        panic!("start_coord length must match the second dimension of points")
    }

    let mut res_selected_coord = Some(start_coord);
    let mut dist_pts_to_selected_min = Array1::<f32>::from_elem(p, f32::INFINITY);
    let mut selected_pts_coords = Vec::with_capacity(n_samples);

    // Initialize the distances with respect to the start_coord
    let start_dist = points.map_axis(Axis(1), |point| {
        point
            .iter()
            .zip(res_selected_coord.as_ref().unwrap().iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
    });

    Zip::from(&mut dist_pts_to_selected_min)
        .and(&start_dist)
        .for_each(|x, &y| *x = y);

    while selected_pts_coords.len() < n_samples {
        // Select the point with max distance
        let max_idx = dist_pts_to_selected_min
            .indexed_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        let max_coord = points.row(max_idx).to_owned();
        selected_pts_coords.push(max_coord.clone()); // Clone max_coord here
        res_selected_coord = Some(max_coord); // No need to clone again as we've already cloned it

        // Update distance
        let dist = points.map_axis(Axis(1), |point| {
            point
                .iter()
                .zip(res_selected_coord.as_ref().unwrap().iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
        });

        // Update min distance
        Zip::from(&mut dist_pts_to_selected_min)
            .and(&dist)
            .for_each(|x, &y| {
                if *x > y {
                    *x = y;
                }
            });
    }

    Array2::from_shape_vec(
        (n_samples, c),
        selected_pts_coords
            .into_iter()
            .flat_map(|x| x.into_raw_vec())
            .collect(),
    )
    .unwrap()
}

#[pyfunction]
#[pyo3(name = "_fps_sampling")]
fn fps_sampling_py<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    n_samples: usize,
    start_coord: PyReadonlyArray1<f32>,
) -> PyResult<&'py PyArray2<f32>> {
    // Verify that the number of samples is less than the number of points
    if n_samples > points.shape()[0] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_samples must be less than the number of points.",
        ));
    }

    // Ensure that the length of start_coord matches the second dimension of points
    if start_coord.len() != points.shape()[1] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Length of start_coord must match the second dimension of points.",
        ));
    }

    let points_array = points.as_array();
    let start_coord_array = start_coord.as_array();

    let coords =
        py.allow_threads(|| fps_sampling(points_array, n_samples, start_coord_array.to_owned()));
    let ret = coords.to_pyarray(py);
    Ok(ret)
}

fn k_means(
    data_set: Array2<f32>,
    mut centroids: Array2<f32>,
    max_iterations: usize,
    tolerance: f32,
) -> Array2<f32> {
    for _ in 0..max_iterations {
        let labels = get_labels(&data_set, &centroids);
        let new_centroids = get_centroids(&data_set, &labels, centroids.nrows());

        // Calculate the total change in centroids
        let centroid_shift: f32 = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Update centroids
        centroids = new_centroids;

        // Check if the change is below the tolerance threshold
        if centroid_shift < tolerance {
            break;
        }
    }

    centroids
}

fn get_labels(data_set: &Array2<f32>, centroids: &Array2<f32>) -> Array1<usize> {
    data_set.map_axis(Axis(1), |point| {
        centroids
            .outer_iter()
            .enumerate()
            .map(|(idx, centroid)| (idx, (&point - &centroid).mapv(|a| a * a).sum()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    })
}

fn get_centroids(data_set: &Array2<f32>, labels: &Array1<usize>, k: usize) -> Array2<f32> {
    let mut centroids = Array2::<f32>::zeros((k, data_set.ncols()));
    let mut counts = Array1::<usize>::zeros(k);

    for (idx, &label) in labels.iter().enumerate() {
        let mut centroid_row = centroids.row_mut(label);
        let data_point = data_set.row(idx);
        centroid_row.zip_mut_with(&data_point, |x, &y| *x += y);
        counts[label] += 1;
    }

    for (i, mut row) in centroids.outer_iter_mut().enumerate() {
        if counts[i] > 0 {
            row.mapv_inplace(|x| x / counts[i] as f32);
        }
    }

    centroids
}

#[pyfunction]
#[pyo3(name = "_k_means")]
fn k_means_py<'py>(
    py: Python<'py>,
    data_set: PyReadonlyArray2<f32>,
    initial_centroids: PyReadonlyArray2<f32>,
    max_iterations: usize,
    tolerance: f32, // Added tolerance parameter
) -> PyResult<&'py PyArray2<f32>> {
    let data_set_owned: Array2<f32> = data_set.as_array().to_owned();
    let initial_centroids_owned: Array2<f32> = initial_centroids.as_array().to_owned();

    // Pass the tolerance parameter to the k_means function
    let centroids = py.allow_threads(|| {
        k_means(
            data_set_owned,
            initial_centroids_owned,
            max_iterations,
            tolerance,
        )
    });

    Ok(centroids.to_pyarray(py))
}

////////////////////////
//                    //
//    Python Entry    //
//                    //
////////////////////////
#[pymodule]
fn pctools(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(k_means_py, m)?)?;
    m.add_function(wrap_pyfunction!(fps_sampling_py, m)?)?;
    Ok(())
}
