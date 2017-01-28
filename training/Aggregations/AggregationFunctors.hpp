namespace AggregationFunctors
{

//---------------------------------------------------------------------------
//Compare timestamps, which is to ensure that only entries created AFTER the reference date are used

struct CompareTimestamps : public thrust::unary_function<float, float>
{

    const float time_stamp_output;

    CompareTimestamps(
        const float _time_stamp_output) : time_stamp_output(_time_stamp_output) {}

    __device__ float operator()(const float &_time_stamp_input) const
    {

        return ((_time_stamp_input > time_stamp_output) ? (0.f) : (1.f));
    }
};
}