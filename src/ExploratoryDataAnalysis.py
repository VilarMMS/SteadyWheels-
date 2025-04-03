import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

class EDA:
    """
    A class for performing Exploratory Data Analysis on encoded datasets
    and handling missing values intelligently based on feature distributions.
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the EDA class with a pandas DataFrame.
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataset to analyze (expected shape: 50k x 120 features)
        """
        self.df = dataframe
        self.na_values = ["", "na", "NA", np.nan, None]
        self.numeric_features = []
        self.categorical_features = []
        self._identify_feature_types()
        
    def _identify_feature_types(self):
        """Identify numeric, categorical, and binary features in the dataset."""
        self.binary_features = []  # New list for binary features
        
        for col in self.df.columns:
            # Check if column has non-numeric values after removing NA values
            clean_col = self.df[col].replace(self.na_values, np.nan).dropna()
            
            try:
                # Convert to numeric
                numeric_values = pd.to_numeric(clean_col)
                
                # Check if binary (contains only 0 and 1)
                unique_values = set(numeric_values.unique())
                if unique_values.issubset({0, 1}) and len(unique_values) <= 2:
                    self.binary_features.append(col)
                else:
                    self.numeric_features.append(col)
                    
            except:
                self.categorical_features.append(col)
                
    def get_basic_info(self):
        """Get basic information about the dataset."""
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Number of Numeric Features: {len(self.numeric_features)}")
        print(f"Number of Categorical Features: {len(self.categorical_features)}")
        
        # Missing values analysis
        missing_data = self.df.replace(self.na_values, np.nan).isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_summary = pd.DataFrame({
            'Missing Values': missing_data,
            'Percentage': missing_percent
        }).sort_values('Percentage', ascending=False)
        
        print("\nTop 10 Features with Missing Values:")
        print(missing_summary.head(10))
        
        return missing_summary
    
    def analyze_numeric_features(self, n_features=10):
        """
        Analyze the distribution of numeric features.
        
        Parameters:
        -----------
        n_features : int
            Number of features to analyze (default: 10)
        
        Returns:
        --------
        pd.DataFrame
            Summary statistics for numeric features
        """
        if not self.numeric_features:
            print("No numeric features found.")
            return None
            
        # Select top n features with most non-null values
        non_null_counts = self.df[self.numeric_features].replace(self.na_values, np.nan).count()
        top_features = non_null_counts.sort_values(ascending=False).index[:n_features]
        
        # Convert to numeric explicitly
        numeric_df = self.df[top_features].replace(self.na_values, np.nan)
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        
        # Calculate statistics
        stats = numeric_df.describe().T
        stats['skew'] = numeric_df.skew()
        stats['missing_pct'] = (numeric_df.isnull().sum() / len(numeric_df)) * 100
        
        print(f"\nTop {n_features} Numeric Feature Statistics:")
        print(stats)
        
        # Plot distributions
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(top_features[:min(6, len(top_features))]):
            plt.subplot(2, 3, i+1)
            sns.histplot(numeric_df[feature].dropna(), kde=True)
            plt.title(f'Distribution of {feature}')
            plt.tight_layout()
        plt.show()
        
        return stats
    
    def analyze_categorical_features(self, n_features=10, top_categories=5):
        """
        Analyze categorical features.
        
        Parameters:
        -----------
        n_features : int
            Number of features to analyze (default: 10)
        top_categories : int
            Number of top categories to display per feature (default: 5)
            
        Returns:
        --------
        dict
            Dictionary with categorical feature statistics
        """
        if not self.categorical_features:
            print("No categorical features found.")
            return None
            
        # Select top n features with most non-null values
        non_null_counts = self.df[self.categorical_features].replace(self.na_values, np.nan).count()
        top_features = non_null_counts.sort_values(ascending=False).index[:n_features]
        
        results = {}
        for feature in top_features:
            # Get value counts excluding NA values
            clean_values = self.df[feature].replace(self.na_values, np.nan).dropna()
            value_counts = clean_values.value_counts().head(top_categories)
            
            # Calculate statistics
            results[feature] = {
                'unique_values': len(clean_values.unique()),
                'missing_pct': (self.df[feature].replace(self.na_values, np.nan).isnull().sum() / len(self.df)) * 100,
                'top_values': value_counts.to_dict()
            }
            
            print(f"\nFeature: {feature}")
            print(f"Unique Values: {results[feature]['unique_values']}")
            print(f"Missing: {results[feature]['missing_pct']:.2f}%")
            print(f"Top {top_categories} Categories:")
            for val, count in results[feature]['top_values'].items():
                print(f"  - {val}: {count} ({count/len(self.df)*100:.2f}%)")
        
        return results
    
    def run_pca_analysis(self, n_components=2):
        """
        Run PCA on numeric features to identify patterns and outliers.
        
        Parameters:
        -----------
        n_components : int
            Number of principal components to compute (default: 2)
            
        Returns:
        --------
        np.ndarray
            PCA components
        """
        if len(self.numeric_features) < 2:
            print("Not enough numeric features for PCA.")
            return None
            
        # Convert all numeric columns to float
        numeric_df = self.df[self.numeric_features].replace(self.na_values, np.nan)
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        
        # Fill missing values temporarily for PCA
        numeric_df = numeric_df.fillna(numeric_df.median())
        
        # Standardize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Plot results
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.3)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA of Numeric Features')
        plt.show()
        
        print(f"\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
        print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.2%}")
        
        return pca_result
    
    def feature_correlation_analysis(self, threshold=0.7, n_features=20, method='spearman'):
        """
        Analyze correlations between numeric features using Spearman rank correlation.
        
        Parameters:
        -----------
        threshold : float
            Correlation threshold to highlight (default: 0.7)
        n_features : int
            Number of features to include in the correlation matrix (default: 20)
        method : str
            Correlation method: 'spearman' (default) or 'pearson'
            
        Returns:
        --------
        pd.DataFrame
            Correlation matrix
        """
        if len(self.numeric_features) < 2:
            print("Not enough numeric features for correlation analysis.")
            return None
            
        # Convert numeric features to float
        numeric_df = self.df[self.numeric_features].replace(self.na_values, np.nan)
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        
        # Select top n features based on non-null counts
        non_null_counts = numeric_df.count()
        top_features = non_null_counts.sort_values(ascending=False).index[:n_features]
        
        # Calculate correlation matrix using specified method
        corr_matrix = numeric_df[top_features].corr(method=method)
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = 'coolwarm'
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap=cmap, 
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
        
        plt.title(f'Feature Correlation Matrix ({method.capitalize()} Method)')
        plt.tight_layout()
        plt.show()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            print(f"\nHighly Correlated Feature Pairs (|r| >= {threshold}) using {method.capitalize()} method:")
            for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"  - {feat1} & {feat2}: {corr:.3f}")
                
            # Add visualization of top correlated pairs
            if len(high_corr_pairs) > 0:
                n_pairs = min(5, len(high_corr_pairs))
                top_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:n_pairs]
                
                fig, axes = plt.subplots(1, n_pairs, figsize=(n_pairs*4, 4))
                if n_pairs == 1:
                    axes = [axes]
                    
                for i, (feat1, feat2, corr) in enumerate(top_pairs):
                    ax = axes[i]
                    scatter_data = numeric_df[[feat1, feat2]].dropna()
                    ax.scatter(scatter_data[feat1], scatter_data[feat2], alpha=0.5)
                    ax.set_title(f"{feat1} vs {feat2}\n{method.capitalize()} corr: {corr:.3f}")
                    ax.set_xlabel(feat1)
                    if i == 0:
                        ax.set_ylabel(feat2)
                
                plt.tight_layout()
                plt.show()
        else:
            print(f"\nNo feature pairs with correlation >= {threshold} found using {method.capitalize()} method.")
        
        return corr_matrix
        
    def drop_correlated_features(self, threshold=0.7, n_features=20, method='spearman', keep_strategy='higher_variance'):
        """
        Identify and drop one feature from each highly correlated pair.
        
        Parameters:
        -----------
        threshold : float
            Correlation threshold to identify highly correlated pairs (default: 0.7)
        n_features : int
            Number of features to include in the correlation analysis (default: 20)
        method : str
            Correlation method: 'spearman' (default) or 'pearson'
        keep_strategy : str
            Strategy to determine which feature to keep from correlated pairs:
            - 'higher_variance': Keep the feature with higher variance (default)
            - 'first': Keep the first feature alphabetically
            - 'lower_missing': Keep the feature with fewer missing values
            
        Returns:
        --------
        tuple
            (pd.DataFrame, list): Tuple containing:
                - DataFrame with correlated features dropped
                - List of feature names that were dropped
        """
        # Run the correlation analysis first
        corr_matrix = self.feature_correlation_analysis(
            threshold=threshold, 
            n_features=n_features, 
            method=method
        )
        
        if corr_matrix is None:
            return self.df.copy(), []
        
        # Identify highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
        
        if not high_corr_pairs:
            print(f"No feature pairs with correlation >= {threshold} found. No features to drop.")
            return self.df.copy(), []
        
        # Track features to drop
        features_to_drop = set()
        numeric_df = self.df[self.numeric_features].replace(self.na_values, np.nan)
        
        # Convert columns to numeric
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        
        # Process each pair and decide which feature to drop
        print(f"\nDropping one feature from each highly correlated pair (|r| >= {threshold}):")
        
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            # Skip if both features are already marked for dropping
            if feat1 in features_to_drop and feat2 in features_to_drop:
                continue
                
            # Skip if one feature is already marked for dropping
            if feat1 in features_to_drop:
                continue
            if feat2 in features_to_drop:
                continue
            
            # Decide which feature to drop based on the selected strategy
            if keep_strategy == 'higher_variance':
                var1 = numeric_df[feat1].var(skipna=True)
                var2 = numeric_df[feat2].var(skipna=True)
                drop_feature = feat2 if var1 >= var2 else feat1
                keep_feature = feat1 if var1 >= var2 else feat2
                reason = "lower variance"
            
            elif keep_strategy == 'first':
                # Keep the first feature alphabetically
                if feat1 < feat2:
                    drop_feature = feat2
                    keep_feature = feat1
                else:
                    drop_feature = feat1
                    keep_feature = feat2
                reason = "alphabetical order"
            
            elif keep_strategy == 'lower_missing':
                # Keep the feature with fewer missing values
                missing1 = numeric_df[feat1].isna().sum()
                missing2 = numeric_df[feat2].isna().sum()
                drop_feature = feat2 if missing1 <= missing2 else feat1
                keep_feature = feat1 if missing1 <= missing2 else feat2
                reason = "more missing values"
            
            else:
                raise ValueError("Invalid keep_strategy. Choose from 'higher_variance', 'first', or 'lower_missing'")
            
            # Add the selected feature to the drop list
            features_to_drop.add(drop_feature)
            print(f"  - Keeping '{keep_feature}' and dropping '{drop_feature}' ({reason}, correlation: {corr:.3f})")
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_filtered = self.df.copy()
        
        # Drop the features
        features_dropped = list(features_to_drop)
        if features_dropped:
            df_filtered = df_filtered.drop(columns=features_dropped)
            
            # Update feature lists in the class instance
            self.numeric_features = [f for f in self.numeric_features if f not in features_dropped]
            
            print(f"\nDropped {len(features_dropped)} features due to high correlation.")
        
        return df_filtered, features_dropped
            
    def fill_missing_values(self):
        """
        Fill missing values based on feature distributions.
        
        For numeric features:
        - Use mean for normally distributed data
        - Use median for skewed data
        
        For categorical features:
        - Use mode for features with a dominant category
        - Use a special 'Missing' category for balanced distributions
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with filled missing values
        """
        filled_df = self.df.copy()
        fill_methods = {}
        
        print("\nFilling Missing Values:")
        
        # Process numeric features
        for col in self.numeric_features:
            # Convert to numeric
            filled_df[col] = pd.to_numeric(filled_df[col].replace(self.na_values, np.nan), errors='coerce')
            
            if filled_df[col].isnull().sum() > 0:
                # Calculate skewness
                skewness = filled_df[col].dropna().skew()
                
                if abs(skewness) < 0.5:  # Relatively normal distribution
                    fill_value = filled_df[col].mean()
                    method = "mean"
                else:  # Skewed distribution
                    fill_value = filled_df[col].median()
                    method = "median"
                
                filled_df[col] = filled_df[col].fillna(fill_value)
                fill_methods[col] = {
                    'method': method,
                    'value': fill_value,
                    'skewness': skewness
                }
                print(f"  - {col}: Used {method} ({fill_value:.3f}) due to skewness of {skewness:.3f}")
        
        # Process categorical features
        for col in self.categorical_features:
            missing_mask = filled_df[col].isin(self.na_values) | filled_df[col].isna()
            
            if missing_mask.sum() > 0:
                # Get value counts
                value_counts = filled_df.loc[~missing_mask, col].value_counts()
                
                if len(value_counts) > 0:
                    # Check if there's a dominant category (>50%)
                    most_common = value_counts.index[0]
                    most_common_pct = value_counts.iloc[0] / value_counts.sum()
                    
                    if most_common_pct > 0.5:
                        fill_value = most_common
                        method = "mode"
                    else:
                        fill_value = "Missing"
                        method = "special category"
                        
                    filled_df.loc[missing_mask, col] = fill_value
                    fill_methods[col] = {
                        'method': method,
                        'value': fill_value,
                        'most_common_pct': most_common_pct if most_common_pct else None
                    }
                    print(f"  - {col}: Used '{method}' ('{fill_value}') " + 
                          (f"with {most_common_pct:.1%} dominance" if most_common_pct > 0.5 else "for balanced distribution"))
        
        return filled_df, fill_methods
    
    def generate_concise_report(self, title: str):
        """Generate a concise summary report of the dataset."""
        # Basic info
        print("=" * 50)
        print(f"SUMMARY REPORT: {title}")
        print("=" * 50)
        
        # Shape and missing values
        print(f"\nDataset Shape: {self.df.shape[0]:,} samples × {self.df.shape[1]} features")
        
        # Feature types
        print(f"Feature Types: {len(self.numeric_features)} numeric, {len(self.categorical_features)} categorical")
        
        # Missing values
        missing_data = self.df.replace(self.na_values, np.nan).isnull().sum()
        total_missing = missing_data.sum()
        total_cells = self.df.size
        
        print(f"\nMissing Values: {total_missing:,} ({total_missing/total_cells:.2%} of all data)")
        
        # Features with highest missing values
        top_missing = missing_data[missing_data > 0].sort_values(ascending=False).head(5)
        if not top_missing.empty:
            print("\nTop 5 Features with Missing Values:")
            for col, count in top_missing.items():
                print(f"  - {col}: {count:,} ({count/len(self.df):.2%})")
        
        # Numeric feature statistics
        if self.numeric_features:
            numeric_df = self.df[self.numeric_features].replace(self.na_values, np.nan)
            for col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
                
            # Get basic statistics for numeric features
            stats = numeric_df.describe().T
            stats['skew'] = numeric_df.skew()
            
            # Find most skewed and least skewed features
            most_skewed = stats['skew'].dropna().abs().sort_values(ascending=False).head(3)
            
            print("\nNumeric Feature Insights:")
            for col, skew in most_skewed.items():
                direction = "right-skewed" if skew > 0 else "left-skewed"
                print(f"  - {col}: Highly {direction} (skew={skew:.2f})")
        
        # Categorical feature insights
        if self.categorical_features:
            print("\nCategorical Feature Insights:")
            
            for col in self.categorical_features[:3]:  # Limit to top 3 categorical features
                values = self.df[col].replace(self.na_values, np.nan).dropna()
                if len(values) > 0:
                    unique_count = len(values.unique())
                    top_value = values.value_counts().index[0] if not values.empty else "N/A"
                    top_value_pct = values.value_counts().iloc[0] / len(values) if not values.empty else 0
                    
                    cardinality = "high" if unique_count > 100 else "medium" if unique_count > 10 else "low"
                    
                    print(f"  - {col}: {unique_count} unique values ({cardinality} cardinality)")
                    if top_value_pct > 0.5:
                        print(f"    Dominant value: '{top_value}' ({top_value_pct:.1%})")
        
        print("\nRecommended Next Steps:")
        print("  1. Address missing values (use fill_missing_values method)")
        print("  2. Check for outliers in highly skewed numeric features")
        print("  3. Consider encoding high-cardinality categorical features")
        print("  4. Investigate correlation between features")
        
        print("\nNote: Run analyze_numeric_features() and analyze_categorical_features()")
        print("      for more detailed feature-level insights.")
        print("=" * 50)

    def tsne_visualization(self, df: pd.DataFrame, perplexity=120, learning_rate=500, random_state=42):

        from sklearn.manifold import TSNE

        if df is not None:
            self.df = df

        """Perform t-SNE visualization on self.df with color-coded classes."""
        # Ensure 'class' is present
        if 'class' not in self.df.columns:
            raise ValueError("Column 'class' is missing from DataFrame.")

        # Extract features (all columns except 'class')
        features = self.df.drop(columns=['class'])

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state)
        tsne_results = tsne.fit_transform(features)

        # Plot the t-SNE results
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=self.df['class'], palette={'pos': 'red', 'neg': 'blue'}, alpha=0.7)
        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Class')
        plt.show()
