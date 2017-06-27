//
// Created by spades on 21/02/17.
//

#ifndef CGMAPPING_MAP_H
#define CGMAPPING_MAP_H

namespace cgmapping {

    /**
     *
     * @tparam T class type of the map used
     * @tparam C class type used by the map to be updated
     *
     */
    template <class T, class C>
    class map {

        T map_;

    public:
        map(){};

        T _getMap() const {
            return map_;
        }

        inline void _setMap(T& map_) {
            map::map_ = map_;
        }

        inline bool operator==(const map<T, C> &rhs) const {
            return map_ == rhs._getMap();
        }

        inline bool operator!=(const map<T, C> &rhs) const {
            return !(rhs == *this);
        }

        inline map<T,C>& operator=(const map<T,C>& rhs){

            if(this != &rhs)
                map_ = rhs._getMap();

            return *this;
        }

        inline virtual bool _update_map(const C& delta_map) = 0;

    };

}

#endif //CGMAPPING_MAP_H
