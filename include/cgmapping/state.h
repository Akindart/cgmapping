//
// Created by spades on 20/02/17.
//

#ifndef CGMAPPING_STATE_H
#define CGMAPPING_STATE_H

namespace cgmapping {

    template <class T>
    class state {

        T state_;

    public:

        state(){};

        inline T _getStateInfo() const {
            return state_;
        };

        inline void _setStateInfo(const T& state_){
           state::state_ = state_;
        }

        inline bool operator==(const state<T> &rhs) const {
            return state_ == rhs._getStateInfo();
        }

        inline bool operator!=(const state<T> &rhs) const {
            return !(rhs == *this);
        }

        inline state<T>& operator=(const state<T>& rhs){

            if(this != &rhs)
                state_ = rhs._getStateInfo();

            return *this;
        }

        /***
         *
         * @param delta_state it is the update information needed by the state to update itself
         * @return
         *
         */
        virtual bool _update_state(const state<T>& delta_state) = 0;

    };

}

#endif //CGMAPPING_STATE_H
