d3(o4).
d3(o3).
d0(o1).
d2(o1).
d1(o2).

d2(X) :- not d0(X), d1(X).
d1(X) :- not d0(X), d3(X).