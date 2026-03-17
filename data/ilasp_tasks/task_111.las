#constant(obj, o3).
#constant(obj, o4).

#modeh(d3(var(obj))).

#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).

#pos(eg1, {
  d3(o4)
}, {
  d3(o3)
}, {
  d1(o3).
  d1(o4).
  d2(o4).
  d4(o4).
}).