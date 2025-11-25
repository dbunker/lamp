d2(o1).
d1(o4).
d3(o3).
d1(o3).

d2(X) :- not d1(X), d0(X).
d1(X) :- not d0(X), d2(X).
d0(X) :- not d2(X), d3(X).