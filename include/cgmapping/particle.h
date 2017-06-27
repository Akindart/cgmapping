//
// Created by spades on 20/02/17.
//

#ifndef CGMAPPING_PARTICLE_H
#define CGMAPPING_PARTICLE_H

#include <iostream>

namespace cgmapping {

    template<class TState, class TMap>
    class particle {

        TState state_;
        TMap map_;
        double weight_;

    public:

        particle(){};

        inline TState& _getState() const {
            return state_;
        }

        inline void _setState(TState& state) {
            particle::state_ = state;
        }

        inline TMap& _getMap() const {
            return map_;
        }

        inline void _setMap(TMap& map) {
            particle::map_ = map;
        }

        inline double _getWeight() const {
            return weight_;
        }

        inline void _setWeight(double weight) {
            particle::weight_ = weight;
        }

        inline bool operator==(const particle &rhs) const {
            return state_ == rhs._getState() &&
                   map_ == rhs._getMap() &&
                   weight_ == rhs._getWeight();
        }

        inline bool operator!=(const particle &rhs) const {
            return !(rhs == *this);
        }

        inline particle<TState, TMap>& operator=(const particle& rhs){

            if(this != &rhs) {

                state_ = rhs._getState();
                map_ = rhs._getMap();
                weight_ = rhs._getWeight();

            }

            return *this;
        }

    };

}
#endif //CGMAPPING_PARTICLE_H
