pub fn cartesian_product(dims: &[(String, Vec<i64>)]) -> Vec<Vec<(String, i64)>> {
    dims.iter()
        .fold(vec![Vec::new()], |combos, (name, choices)| {
            combos
                .into_iter()
                .flat_map(|combo| {
                    choices.iter().map(move |&choice| {
                        let mut combo = combo.clone();
                        combo.push((name.clone(), choice));
                        combo
                    })
                })
                .collect()
        })
}

#[cfg(test)]
mod tests {
    use super::cartesian_product;

    fn dim(name: &str, choices: &[i64]) -> (String, Vec<i64>) {
        (name.to_string(), choices.to_vec())
    }

    #[test]
    fn empty_space_is_one_empty_combination() {
        assert_eq!(cartesian_product(&[]), vec![Vec::<(String, i64)>::new()]);
    }

    #[test]
    fn counts_the_full_product() {
        let space = [dim("A", &[1, 2, 3]), dim("B", &[10, 20]), dim("C", &[0])];
        let all = cartesian_product(&space);
        #[allow(clippy::identity_op)]
        let want = 3 * 2 * 1;
        assert_eq!(all.len(), want);
        for combo in &all {
            assert_eq!(
                combo.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
                vec!["A", "B", "C"]
            );
        }
    }

    #[test]
    fn last_dimension_varies_fastest() {
        let space = [dim("A", &[1, 2]), dim("B", &[10, 20])];
        let all = cartesian_product(&space);
        let values: Vec<Vec<i64>> = all
            .iter()
            .map(|c| c.iter().map(|(_, v)| *v).collect())
            .collect();
        assert_eq!(
            values,
            vec![vec![1, 10], vec![1, 20], vec![2, 10], vec![2, 20]]
        );
    }
}
