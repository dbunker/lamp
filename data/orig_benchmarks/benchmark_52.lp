d1(o4).
d2(o2).
d3(o2).
d0(o0).

d0(X) :- not d2(X), d1(X), d3(X).
d0(X) :- not d1(X), d2(X), d3(X).
d0(X) :- not d2(X), d1(X), d3(X).