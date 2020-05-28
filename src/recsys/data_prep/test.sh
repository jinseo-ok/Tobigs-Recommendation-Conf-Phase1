echo "Run Data Prep"
echo "python extract_hotel_dense_features.py"
python extract_hotel_dense_features.py
echo "python extract_item_prices.py"
python extract_item_prices.py
echo "python extract_item_prices_rank.py"
python extract_item_prices_rank.py
echo "python generate_click_indices.py"
python generate_click_indices.py
echo "python assign_poi_to_items.py" 
python assign_poi_to_items.py
echo "python extract_city_prices_percentiles.py"
python extract_city_prices_percentiles.py
echo "python extract_item_rating.py"
python extract_item_rating.py
