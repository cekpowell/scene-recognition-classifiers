package uk.ac.soton.ecs.group;

/**
 * Represents a pair of elements.
 * 
 * @author Charles Powell (cp6g18)
 */
public class Tuple<T, S> {

    // member variables
    T first;
    S second;

    /**
     * Class constructor.
     * 
     * @param first The first element in the tuple.
     * @param second The second element in the tuple.
     */
    public Tuple(T first, S second){
        this.first = first;
        this.second = second;
    }

    @Override
    public String toString() {
        return first.toString() + " " + second.toString();
    }

    /////////////////////////
    // GETTERS AND SETTERS //
    /////////////////////////

    public T getFirst(){
        return this.first;
    }

    public S getSecond(){
        return this.second;
    }
}